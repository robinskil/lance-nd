//! Write nd record batches in several calls and commit them once.

use std::collections::{HashSet, VecDeque};
use std::pin::pin;
use std::sync::Arc;

use arrow::array::{RecordBatch, RecordBatchIterator};
use arrow::datatypes::{Schema, SchemaRef};
use futures::Stream;
use lance::Dataset;
use lance::dataset::fragment::write::FragmentCreateBuilder;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{CommitBuilder, NewColumnTransform, WriteMode, WriteParams};
use lance::table::format::Fragment;
use lance_core::datatypes::{OnMissing, OnTypeMismatch, Schema as LanceSchema};
use nd_arrow_array::NdRecordBatch;
use tokio::task::JoinHandle;

use crate::cast::Casting;
use crate::encode::{RowDecoder, encode_rows};
use crate::error::{Error, Result};
use crate::format::check_format;
use crate::options::ColumnOptions;
use crate::schema::{concat_conformed, holds_schema, merge_schemas, missing_fields, used_fields};
use crate::stats::{STATS_PREFIX, is_stats_name};
use crate::store::{StorageOptions, open_dataset, store_params};
use crate::widen::{promotions, widen};
use crate::write::{
    FILE_VERSION, WriteOptions, next_group, prepare_batch, table_defaults, table_has_stats,
};

/// A writer that makes all its rows visible in one commit.
///
/// [`NdWriter::write`] buffers rows. When the buffer has about
/// [`WriteOptions::target_fragment_bytes`] on disk, as estimated from the
/// fragments written so far, a Tokio task writes it as one
/// Lance fragment, and up to [`WriteOptions::write_parallelism`] fragments
/// are written at the same time. Readers do not see the fragments.
/// [`NdWriter::finish`] writes the rest and commits all of them in one new
/// table version, in the order of the rows. The writer must run on a Tokio
/// runtime.
///
/// An append writer adds new fields to the table before it writes the first
/// fragment with them, so readers can see new empty columns before
/// [`NdWriter::finish`]. `finish` then commits a plain append.
///
/// A writer that is aborted or dropped commits nothing, and stops its
/// fragment writes. Its data files stay until a Lance cleanup removes them. One exception: under
/// [`Casting::Promote`], a write that widens the table first commits the
/// rows written before it.
pub struct NdWriter {
    uri: String,
    /// The table at the start, for an append or an overwrite.
    base: Option<Dataset>,
    /// `true` when `finish` replaces the table.
    overwrite: bool,
    defaults: ColumnOptions,
    /// `true` when the rows get stats.
    stats: bool,
    options: WriteOptions,
    /// The table schema with all staged fields, as Arrow and as Lance.
    schema: Option<(Schema, LanceSchema)>,
    /// Rows for the next fragment, and their size in memory.
    buffer: Vec<RecordBatch>,
    buffered_bytes: usize,
    /// Fragment writes that run, in the order of their rows, with the
    /// size of their rows in memory.
    pending: VecDeque<(JoinHandle<Result<Vec<Fragment>>>, usize)>,
    fragments: Vec<Fragment>,
    /// The bytes in memory and on disk of the fragments written so far.
    written: DiskRatio,
    /// The error of a failed fragment write. `finish` then fails too.
    failed: Option<Arc<Error>>,
}

impl NdWriter {
    /// A writer for a new table at `uri`.
    ///
    /// `options.mode` is [`WriteMode::Create`], which fails when the table
    /// exists, or [`WriteMode::Overwrite`], which replaces the table at
    /// [`NdWriter::finish`].
    pub async fn create(uri: &str, options: &WriteOptions) -> Result<Self> {
        let existing = open_existing(uri, &options.storage_options).await?;
        let overwrite = match options.mode {
            WriteMode::Create if existing.is_some() => {
                return Err(Error::TableExists(uri.to_string()));
            }
            WriteMode::Create => false,
            WriteMode::Overwrite => true,
            WriteMode::Append => {
                return Err(Error::InvalidArgument(
                    "use NdWriter::append to append to a table".to_string(),
                ));
            }
        };
        Ok(Self {
            uri: uri.to_string(),
            base: existing,
            overwrite,
            defaults: options.column_defaults.clone().unwrap_or_default(),
            stats: options.stats,
            options: options.clone(),
            schema: None,
            buffer: Vec::new(),
            buffered_bytes: 0,
            pending: VecDeque::new(),
            fragments: Vec::new(),
            written: DiskRatio::default(),
            failed: None,
        })
    }

    /// A writer that appends to the table at `uri`. Fails when the table
    /// does not exist. `options.mode` is ignored.
    pub async fn append(uri: &str, options: &WriteOptions) -> Result<Self> {
        let dataset = open_existing(uri, &options.storage_options)
            .await?
            .ok_or_else(|| Error::TableNotFound(uri.to_string()))?;
        check_format(&dataset)?;
        let defaults = table_defaults(&dataset, options)?;
        let stats = table_has_stats(&dataset);
        let schema = (Schema::from(dataset.schema()), dataset.schema().clone());
        Ok(Self {
            uri: uri.to_string(),
            base: Some(dataset),
            overwrite: false,
            defaults,
            stats,
            options: options.clone(),
            schema: Some(schema),
            buffer: Vec::new(),
            buffered_bytes: 0,
            pending: VecDeque::new(),
            fragments: Vec::new(),
            written: DiskRatio::default(),
            failed: None,
        })
    }

    /// Write a stream of rows, without a commit. Each nd record batch is one row.
    ///
    /// The writer encodes the rows in groups, see [`WriteOptions`]. A group
    /// that fails a check, for example a type change, is rejected: the
    /// writer returns the error and stays usable, and the groups before it
    /// stay written. A failed fragment write makes the writer fail:
    /// [`NdWriter::finish`] then commits nothing.
    pub async fn write(&mut self, rows: impl Stream<Item = Result<NdRecordBatch>>) -> Result<()> {
        self.check()?;
        let mut rows = pin!(rows);
        while let Some(group) = next_group(&mut rows, &self.options).await? {
            let batch = encode_rows(&group)?;
            self.promote(&batch).await?;
            let table = self.schema.as_ref().map(|(arrow, _)| arrow);
            let batch = prepare_batch(batch, &self.defaults, &self.options, self.stats, table)?;
            self.push(batch).await?;
        }
        Ok(())
    }

    /// Write the rows of an encoded record batch, without a commit.
    ///
    /// `batch` holds nd columns and an optional `__global_attrs__` column,
    /// as [`crate::encode::encode_rows`] gives them. The writer checks the
    /// columns, but it does not decode the rows. A batch that fails a check
    /// is rejected as a whole, and the writer stays usable. A failed
    /// fragment write makes the writer fail.
    pub async fn write_batch(&mut self, batch: RecordBatch) -> Result<()> {
        self.check()?;
        self.write_encoded(batch).await
    }

    fn check(&self) -> Result<()> {
        match &self.failed {
            Some(error) => Err(Error::WriterFailed(error.clone())),
            None => Ok(()),
        }
    }

    fn fail_on(&mut self, result: Result<()>) -> Result<()> {
        result.map_err(|error| {
            let error = Arc::new(error);
            self.failed = Some(error.clone());
            Error::WriterFailed(error)
        })
    }

    async fn write_encoded(&mut self, batch: RecordBatch) -> Result<()> {
        if let Some(field) = batch
            .schema()
            .fields()
            .iter()
            .find(|field| is_stats_name(field.name()))
        {
            return Err(Error::SchemaMismatch(format!(
                "variable name '{}' is reserved, also each name that starts with {STATS_PREFIX}",
                field.name()
            )));
        }
        // Fails for a column that is not an nd column or the global attributes.
        RowDecoder::try_new(&batch)?;
        if batch.num_rows() == 0 {
            return Ok(());
        }
        self.promote(&batch).await?;
        let table = self.schema.as_ref().map(|(arrow, _)| arrow);
        let batch = prepare_batch(batch, &self.defaults, &self.options, self.stats, table)?;
        self.push(batch).await
    }

    /// Widen the variables of `batch` in the table as [`Casting::Promote`]
    /// needs. The staged fragments use the old field ids that a widen
    /// replaces, so the rows written so far are committed first.
    async fn promote(&mut self, batch: &RecordBatch) -> Result<()> {
        if self.options.casting != Casting::Promote {
            return Ok(());
        }
        let Some((arrow, _)) = &self.schema else {
            return Ok(());
        };
        let widenings = promotions(batch, arrow);
        if widenings.is_empty() {
            return Ok(());
        }
        let result = async {
            let staged =
                !(self.buffer.is_empty() && self.pending.is_empty() && self.fragments.is_empty());
            if staged || self.base.is_none() {
                self.commit().await?;
            }
            let dataset = self.base.as_mut().expect("a committed table");
            for (variable, to) in &widenings {
                widen(dataset, variable, to).await?;
            }
            self.schema = Some((Schema::from(dataset.schema()), dataset.schema().clone()));
            Ok(())
        }
        .await;
        self.fail_on(result)
    }

    /// Grow the staged schema with `batch`, then buffer it. A full buffer
    /// becomes a fragment.
    ///
    /// A schema conflict changes nothing, so it only rejects `batch`. A
    /// failed fragment write makes the writer fail.
    async fn push(&mut self, batch: RecordBatch) -> Result<()> {
        // Most rows bring no new field. A merge copies the full staged
        // schema, which is slow with thousands of variables.
        let grows = match &self.schema {
            Some((arrow, _)) => !holds_schema(arrow, &batch.schema()),
            None => true,
        };
        if grows {
            let (arrow, lance) = match &self.schema {
                Some((arrow, lance)) => (
                    merge_schemas(arrow, &batch.schema())?,
                    lance.merge(batch.schema().as_ref())?,
                ),
                None => (
                    batch.schema().as_ref().clone(),
                    LanceSchema::try_from(batch.schema().as_ref())?,
                ),
            };
            let mut lance = lance;
            lance.set_field_id(self.base_max_field_id());
            self.schema = Some((arrow, lance));
        }
        self.buffered_bytes += batch.get_array_memory_size();
        self.buffer.push(batch);
        if self.written.on_disk(self.buffered_bytes) >= self.options.target_fragment_bytes as f64 {
            let result = self.flush().await;
            return self.fail_on(result);
        }
        Ok(())
    }

    /// Start a fragment write of the buffer. Wait for the oldest write when
    /// too many run.
    async fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        self.evolve().await?;
        let batches = std::mem::take(&mut self.buffer);
        let memory = std::mem::take(&mut self.buffered_bytes);
        let (arrow, lance) = self.schema.as_ref().expect("a buffered row has a schema");
        // Only the columns of the buffered rows: the other columns read as
        // null. Rows of one buffer often share a schema.
        let schemas: Vec<SchemaRef> = batches
            .iter()
            .map(|batch| batch.schema())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let present = Arc::new(used_fields(arrow, &schemas));
        // One batch per fragment: Lance 11 fails to read back a fragment of
        // several batches when a struct column has nulls and large lists.
        let batch = concat_conformed(&present, &batches)?;
        let projected =
            lance.project_by_schema(present.as_ref(), OnMissing::Error, OnTypeMismatch::Error)?;
        let uri = self.uri.clone();
        let store_params = store_params(&self.options.storage_options);
        let task = tokio::spawn(async move {
            let params = WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(FILE_VERSION),
                store_params,
                ..Default::default()
            };
            let reader = RecordBatchIterator::new([Ok(batch)], present);
            Ok(FragmentCreateBuilder::new(&uri)
                .schema(&projected)
                .write_params(&params)
                .write_fragments(reader)
                .await?)
        });
        self.pending.push_back((task, memory));
        while self.pending.len() > self.options.write_parallelism.max(1) {
            self.wait_oldest().await?;
        }
        Ok(())
    }

    /// Add the staged fields that the table lacks to the table, as all-null
    /// columns, and take the field ids of the table. Fragments then always
    /// use the ids of the table.
    ///
    /// Lance validates a Merge commit in a time that grows with the square of
    /// the schema fields, which takes minutes with thousands of variables.
    /// `add_columns` skips that validation.
    async fn evolve(&mut self) -> Result<()> {
        let (Some(base), Some((arrow, _))) = (self.base.as_mut(), self.schema.as_ref()) else {
            return Ok(());
        };
        if self.overwrite {
            return Ok(());
        }
        let missing = missing_fields(Schema::from(base.schema()).fields(), arrow.fields());
        if missing.is_empty() {
            return Ok(());
        }
        // Boxed: the add_columns future is large, and would grow the future
        // of each write beyond the compiler's layout depth.
        Box::pin(base.add_columns(
            NewColumnTransform::AllNulls(Arc::new(Schema::new(missing))),
            None,
            None,
        ))
        .await?;
        self.schema = Some((Schema::from(base.schema()), base.schema().clone()));
        Ok(())
    }

    /// Wait for the oldest fragment write and keep its fragments.
    async fn wait_oldest(&mut self) -> Result<()> {
        let Some((task, memory)) = self.pending.pop_front() else {
            return Ok(());
        };
        let fragments = task.await??;
        self.written.add(memory, &fragments);
        self.fragments.extend(fragments);
        Ok(())
    }

    /// The highest field id of the table that this writer appends to.
    fn base_max_field_id(&self) -> Option<i32> {
        match (&self.base, self.overwrite) {
            (Some(dataset), false) => Some(dataset.manifest.max_field_id()),
            _ => None,
        }
    }

    /// Commit all written rows in one new table version.
    ///
    /// Without rows, a new table or an overwrite fails, and an append returns
    /// the table unchanged.
    pub async fn finish(mut self) -> Result<Dataset> {
        if let Some(error) = &self.failed {
            return Err(Error::WriterFailed(error.clone()));
        }
        self.commit().await
    }

    /// Commit the staged rows. The writer then appends to the new version.
    async fn commit(&mut self) -> Result<Dataset> {
        self.flush().await?;
        while !self.pending.is_empty() {
            self.wait_oldest().await?;
        }
        let Some((_, schema)) = self.schema.clone().filter(|_| !self.fragments.is_empty()) else {
            return match (&self.base, self.overwrite) {
                (Some(dataset), false) => Ok(dataset.clone()),
                _ => Err(Error::InvalidArgument(format!(
                    "no rows to create the table at {}",
                    self.uri
                ))),
            };
        };
        let fragments = std::mem::take(&mut self.fragments);
        let dataset = match (self.base.take(), self.overwrite) {
            (None, overwrite) => {
                // Another writer can create the table after `create`.
                let storage_options = &self.options.storage_options;
                if !overwrite && open_existing(&self.uri, storage_options).await?.is_some() {
                    return Err(Error::TableExists(self.uri.clone()));
                }
                let operation = overwrite_operation(numbered(fragments, 0), schema);
                let mut builder =
                    CommitBuilder::new(self.uri.as_str()).with_storage_format(FILE_VERSION);
                if let Some(params) = store_params(storage_options) {
                    builder = builder.with_store_params(params);
                }
                builder
                    .execute(Transaction::new(0, operation, None))
                    .await?
            }
            (Some(base), true) => {
                let version = base.version().version;
                let operation = overwrite_operation(numbered(fragments, 0), schema);
                CommitBuilder::new(Arc::new(base))
                    .execute(Transaction::new(version, operation, None))
                    .await?
            }
            // No new fields: new rows only, which Lance merges with concurrent appends.
            (Some(base), false) if schema.max_field_id() == base.schema().max_field_id() => {
                commit_rows(base, fragments, &schema).await?
            }
            (Some(base), false) => {
                let version = base.version().version;
                let mut all: Vec<Fragment> = base
                    .get_fragments()
                    .iter()
                    .map(|f| f.metadata().clone())
                    .collect();
                let first_id = base.manifest.max_fragment_id().map_or(0, |id| id + 1);
                all.extend(numbered(fragments, first_id));
                // New fields are nullable, so old rows read them as null.
                let operation = Operation::Merge {
                    fragments: all,
                    schema,
                    preserves_nullability: true,
                };
                CommitBuilder::new(Arc::new(base))
                    .execute(Transaction::new(version, operation, None))
                    .await?
            }
        };
        self.base = Some(dataset.clone());
        self.overwrite = false;
        self.schema = Some((Schema::from(dataset.schema()), dataset.schema().clone()));
        Ok(dataset)
    }

    /// Drop all written rows without a commit. Stops the fragment writes
    /// that run. A drop of the writer does the same.
    pub fn abort(self) {}
}

impl Drop for NdWriter {
    /// Stop the fragment writes that run. Their rows are never committed.
    fn drop(&mut self) {
        for (task, _) in &self.pending {
            task.abort();
        }
    }
}

/// The bytes in memory and on disk of the written fragments, to estimate
/// the size on disk of the buffered rows.
#[derive(Debug, Default)]
struct DiskRatio {
    memory: u64,
    disk: u64,
}

impl DiskRatio {
    /// The lowest ratio of bytes on disk to bytes in memory. It limits the
    /// buffer to 8 times the target, for data that compresses very well.
    const MIN: f64 = 1.0 / 8.0;

    /// Count the rows of `memory` bytes that became `fragments`. A data file
    /// of unknown size counts nothing.
    fn add(&mut self, memory: usize, fragments: &[Fragment]) {
        let files = fragments.iter().flat_map(|fragment| fragment.files.iter());
        let sizes: Option<Vec<u64>> = files
            .map(|file| file.file_size_bytes.get().map(|size| size.get()))
            .collect();
        if let Some(sizes) = sizes {
            self.memory += memory as u64;
            self.disk += sizes.iter().sum::<u64>();
        }
    }

    /// The estimated bytes on disk of `memory` bytes in memory. Before the
    /// first fragment, the bytes in memory.
    fn on_disk(&self, memory: usize) -> f64 {
        let ratio = if self.memory == 0 {
            1.0
        } else {
            (self.disk as f64 / self.memory as f64).max(Self::MIN)
        };
        memory as f64 * ratio
    }
}

/// The most tries of a row commit that a concurrent commit preempts.
const MAX_COMMIT_ATTEMPTS: usize = 10;

/// Commit `fragments` as new rows of `base`, with the fields of `staged`.
///
/// The commit is an `Update` with new fragments only. Lance merges it with
/// concurrent appends, deletes and compactions. A concurrent schema change,
/// for example a widen, makes Lance reject it. Then the writer checks that
/// the table still has each field of the fragments, and tries again.
async fn commit_rows(
    mut base: Dataset,
    fragments: Vec<Fragment>,
    staged: &LanceSchema,
) -> Result<Dataset> {
    for attempt in 1..=MAX_COMMIT_ATTEMPTS {
        check_fields(&base, &fragments, staged)?;
        let operation = Operation::Update {
            removed_fragment_ids: Vec::new(),
            updated_fragments: Vec::new(),
            new_fragments: fragments.clone(),
            fields_modified: Vec::new(),
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: Vec::new(),
            update_mode: None,
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        };
        let version = base.version().version;
        match CommitBuilder::new(Arc::new(base.clone()))
            .execute(Transaction::new(version, operation, None))
            .await
        {
            Ok(dataset) => return Ok(dataset),
            Err(lance::Error::RetryableCommitConflict { .. }) if attempt < MAX_COMMIT_ATTEMPTS => {
                base.checkout_latest().await?;
            }
            Err(error) => return Err(error.into()),
        }
    }
    unreachable!("the last attempt returns")
}

/// Fails when `table` lacks a field that the data files of `fragments` hold.
/// A widen gives a variable new field ids, so rows with the old ids would
/// read as absent.
fn check_fields(table: &Dataset, fragments: &[Fragment], staged: &LanceSchema) -> Result<()> {
    let ids: HashSet<i32> = table.schema().field_ids().into_iter().collect();
    let lost = fragments
        .iter()
        .flat_map(|fragment| fragment.files.iter())
        .flat_map(|file| file.fields.iter())
        .find(|id| **id >= 0 && !ids.contains(*id));
    match lost {
        Some(id) => Err(Error::SchemaMismatch(format!(
            "the table changed or dropped '{}' during the write, nothing is committed",
            staged
                .field_path(*id)
                .unwrap_or_else(|_| format!("field {id}"))
        ))),
        None => Ok(()),
    }
}

/// Give the fragments ids from `first_id` on. Lance assigns ids only for appends.
fn numbered(mut fragments: Vec<Fragment>, first_id: u64) -> Vec<Fragment> {
    for (id, fragment) in (first_id..).zip(&mut fragments) {
        fragment.id = id;
    }
    fragments
}

fn overwrite_operation(fragments: Vec<Fragment>, schema: LanceSchema) -> Operation {
    Operation::Overwrite {
        fragments,
        schema,
        config_upsert_values: None,
        initial_bases: None,
    }
}

/// The table at `uri`, or `None` when it does not exist.
pub(crate) async fn open_existing(uri: &str, options: &StorageOptions) -> Result<Option<Dataset>> {
    match open_dataset(uri, options).await {
        Ok(dataset) => Ok(Some(dataset)),
        Err(lance::Error::DatasetNotFound { .. }) => Ok(None),
        Err(error) => Err(error.into()),
    }
}
