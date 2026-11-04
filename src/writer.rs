//! Write nd record batches in several calls and commit them once.

use std::collections::VecDeque;
use std::pin::pin;
use std::sync::Arc;

use arrow::array::{RecordBatch, RecordBatchIterator};
use arrow::compute::concat_batches;
use arrow::datatypes::Schema;
use futures::Stream;
use lance::Dataset;
use lance::dataset::fragment::write::FragmentCreateBuilder;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{CommitBuilder, WriteMode, WriteParams};
use lance::table::format::Fragment;
use lance_core::datatypes::{OnMissing, OnTypeMismatch, Schema as LanceSchema};
use nd_arrow_array::NdRecordBatch;
use tokio::task::JoinHandle;

use crate::encode::{RowDecoder, encode_rows};
use crate::error::{Error, Result};
use crate::options::ColumnOptions;
use crate::schema::{conform_batch, merge_schemas};
use crate::stats::STATS_FIELD;
use crate::write::{
    FILE_VERSION, WriteOptions, next_group, prepare_batch, table_defaults, table_has_stats,
};

/// A writer that makes all its rows visible in one commit.
///
/// [`NdWriter::write`] buffers rows. When the buffer has about
/// [`WriteOptions::target_fragment_bytes`], a Tokio task writes it as one
/// Lance fragment, and up to [`WriteOptions::write_parallelism`] fragments
/// are written at the same time. Readers do not see the fragments.
/// [`NdWriter::finish`] writes the rest and commits all of them in one new
/// table version, in the order of the rows. The writer must run on a Tokio
/// runtime.
///
/// A writer that is aborted or dropped commits nothing. Its data files stay
/// until a Lance cleanup removes them.
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
    /// Fragment writes that run, in the order of their rows.
    pending: VecDeque<JoinHandle<Result<Vec<Fragment>>>>,
    fragments: Vec<Fragment>,
    /// The error of a failed fragment write. `finish` then fails too.
    failed: Option<String>,
}

impl NdWriter {
    /// A writer for a new table at `uri`.
    ///
    /// `options.mode` is [`WriteMode::Create`], which fails when the table
    /// exists, or [`WriteMode::Overwrite`], which replaces the table at
    /// [`NdWriter::finish`].
    pub async fn create(uri: &str, options: &WriteOptions) -> Result<Self> {
        let existing = open_existing(uri).await?;
        let overwrite = match options.mode {
            WriteMode::Create if existing.is_some() => {
                return Err(Error::InvalidArgument(format!("a table exists at {uri}")));
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
            failed: None,
        })
    }

    /// A writer that appends to the table at `uri`. Fails when the table
    /// does not exist. `options.mode` is ignored.
    pub async fn append(uri: &str, options: &WriteOptions) -> Result<Self> {
        let dataset = open_existing(uri)
            .await?
            .ok_or_else(|| Error::InvalidArgument(format!("no table at {uri}")))?;
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
            Some(error) => Err(Error::InvalidArgument(format!(
                "an earlier write failed: {error}"
            ))),
            None => Ok(()),
        }
    }

    fn fail_on(&mut self, result: Result<()>) -> Result<()> {
        if let Err(error) = &result {
            self.failed = Some(error.to_string());
        }
        result
    }

    async fn write_encoded(&mut self, batch: RecordBatch) -> Result<()> {
        if batch.schema().column_with_name(STATS_FIELD).is_some() {
            return Err(Error::SchemaMismatch(format!(
                "variable name '{STATS_FIELD}' is reserved"
            )));
        }
        // Fails for a column that is not an nd column or the global attributes.
        RowDecoder::try_new(&batch)?;
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let table = self.schema.as_ref().map(|(arrow, _)| arrow);
        let batch = prepare_batch(batch, &self.defaults, &self.options, self.stats, table)?;
        self.push(batch).await
    }

    /// Grow the staged schema with `batch`, then buffer it. A full buffer
    /// becomes a fragment.
    ///
    /// A schema conflict changes nothing, so it only rejects `batch`. A
    /// failed fragment write makes the writer fail.
    async fn push(&mut self, batch: RecordBatch) -> Result<()> {
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
        self.buffered_bytes += batch.get_array_memory_size();
        self.buffer.push(batch);
        if self.buffered_bytes >= self.options.target_fragment_bytes {
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
        let batches = std::mem::take(&mut self.buffer);
        self.buffered_bytes = 0;
        let (arrow, lance) = self.schema.as_ref().expect("a buffered row has a schema");
        // Only the columns of the buffered rows: the other fields read as null.
        let present: Vec<_> = arrow
            .fields()
            .iter()
            .filter(|field| {
                batches
                    .iter()
                    .any(|b| b.schema().column_with_name(field.name()).is_some())
            })
            .cloned()
            .collect();
        let present = Arc::new(Schema::new_with_metadata(present, arrow.metadata().clone()));
        let batches = batches
            .iter()
            .map(|batch| conform_batch(batch, &present))
            .collect::<Result<Vec<_>>>()?;
        // One batch per fragment: Lance 11 fails to read back a fragment of
        // several batches when a struct column has nulls and large lists.
        let batch = concat_batches(&present, &batches)?;
        let projected =
            lance.project_by_schema(present.as_ref(), OnMissing::Error, OnTypeMismatch::Error)?;
        let uri = self.uri.clone();
        self.pending.push_back(tokio::spawn(async move {
            let params = WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(FILE_VERSION),
                ..Default::default()
            };
            let reader = RecordBatchIterator::new([Ok(batch)], present);
            Ok(FragmentCreateBuilder::new(&uri)
                .schema(&projected)
                .write_params(&params)
                .write_fragments(reader)
                .await?)
        }));
        while self.pending.len() > self.options.write_parallelism.max(1) {
            self.wait_oldest().await?;
        }
        Ok(())
    }

    /// Wait for the oldest fragment write and keep its fragments.
    async fn wait_oldest(&mut self) -> Result<()> {
        let Some(task) = self.pending.pop_front() else {
            return Ok(());
        };
        let fragments = task.await.map_err(|error| {
            Error::InvalidArgument(format!("a fragment write failed: {error}"))
        })??;
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
        if let Some(error) = self.failed {
            return Err(Error::InvalidArgument(format!(
                "an earlier write failed, nothing is committed: {error}"
            )));
        }
        self.flush().await?;
        while !self.pending.is_empty() {
            self.wait_oldest().await?;
        }
        let Some((_, schema)) = self.schema.filter(|_| !self.fragments.is_empty()) else {
            return match (self.base, self.overwrite) {
                (Some(dataset), false) => Ok(dataset),
                _ => Err(Error::InvalidArgument(format!(
                    "no rows to create the table at {}",
                    self.uri
                ))),
            };
        };
        let fragments = self.fragments;
        let dataset = match (self.base, self.overwrite) {
            (None, overwrite) => {
                // Another writer can create the table after `create`.
                if !overwrite && open_existing(&self.uri).await?.is_some() {
                    return Err(Error::InvalidArgument(format!(
                        "a table exists at {}",
                        self.uri
                    )));
                }
                let operation = overwrite_operation(numbered(fragments, 0), schema);
                CommitBuilder::new(self.uri.as_str())
                    .with_storage_format(FILE_VERSION)
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
            // No new fields: a plain append, which Lance merges with concurrent appends.
            (Some(base), false) if schema.max_field_id() == base.schema().max_field_id() => {
                let version = base.version().version;
                CommitBuilder::new(Arc::new(base))
                    .execute(Transaction::new(
                        version,
                        Operation::Append { fragments },
                        None,
                    ))
                    .await?
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
        Ok(dataset)
    }

    /// Drop all written rows without a commit. Stops the fragment writes
    /// that run.
    pub fn abort(self) {
        for task in &self.pending {
            task.abort();
        }
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
async fn open_existing(uri: &str) -> Result<Option<Dataset>> {
    match Dataset::open(uri).await {
        Ok(dataset) => Ok(Some(dataset)),
        Err(lance::Error::DatasetNotFound { .. }) => Ok(None),
        Err(error) => Err(error.into()),
    }
}
