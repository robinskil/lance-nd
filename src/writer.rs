//! Write nd record batches in several calls and commit them once.

use std::collections::{HashMap, HashSet, VecDeque};
use std::pin::pin;
use std::sync::Arc;

use arrow::array::{RecordBatch, RecordBatchIterator};
use arrow::datatypes::{FieldRef, Schema};
use futures::{FutureExt, Stream, StreamExt, TryStreamExt, stream};
use lance::Dataset;
use lance::dataset::fragment::write::FragmentCreateBuilder;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{CommitBuilder, WriteMode as LanceWriteMode, WriteParams};
use lance::table::format::Fragment;
use lance_core::datatypes::Schema as LanceSchema;
use nd_arrow_array::NdRecordBatch;
use nd_arrow_array::extension::{GLOBAL_ATTRIBUTES_FIELD, is_global_attributes_field};
use tokio::task::JoinHandle;

use crate::cast::Casting;
use crate::encode::{RowDecoder, encode_rows};
use crate::error::{Error, Result};
use crate::format::check_format;
use crate::options::ColumnOptions;
use crate::schema::{concat_conformed, holds, merge_field};
use crate::stats::{STATS_PREFIX, is_stats_name};
use crate::store::{StorageOptions, open_dataset, store_params};
use crate::widen::{commit_schema, promotions, widen_all};
use crate::write::{
    FILE_VERSION, WriteMode, WriteOptions, next_group, prepare_batch, table_defaults,
    table_has_stats,
};

/// A writer that makes all its rows visible in one commit.
///
/// [`NdWriter::write`] buffers rows. When the buffer has about
/// [`WriteOptions::target_fragment_bytes`] on disk, as estimated from the
/// fragments written so far, a Tokio task writes it as Lance fragments:
/// one per run of rows with similar variables, each with only the columns
/// of its rows. Up to [`WriteOptions::write_parallelism`] tasks run at the
/// same time. Readers do not see the fragments.
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
/// rows written before it. With [`WriteOptions::defer_promotion`], these
/// commits and the widen are detached Lance versions that readers do not
/// see, and `finish` makes the last one the table version.
pub struct NdWriter {
    uri: String,
    /// The table at the start, for an append or an overwrite.
    base: Option<Dataset>,
    /// `true` when `finish` replaces the table.
    overwrite: bool,
    /// The version of the table when the writer starts.
    start_version: Option<u64>,
    /// `true` after a deferred widen: each commit is then a detached version.
    detached: bool,
    defaults: ColumnOptions,
    /// `true` when the rows get stats.
    stats: bool,
    options: WriteOptions,
    /// The table schema with all staged fields, as Arrow and as Lance, with
    /// the fields in the same order.
    schema: Option<(Schema, LanceSchema)>,
    /// The position of each top-level field of `schema`, by name.
    index: HashMap<String, usize>,
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
    #[tracing::instrument(skip_all, fields(uri = %uri, mode = ?options.mode), err)]
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
            start_version: existing.as_ref().map(|dataset| dataset.version().version),
            detached: false,
            base: existing,
            overwrite,
            defaults: options.column_defaults.clone().unwrap_or_default(),
            stats: options.stats,
            options: options.clone(),
            schema: None,
            index: HashMap::new(),
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
    #[tracing::instrument(skip_all, fields(uri = %uri), err)]
    pub async fn append(uri: &str, options: &WriteOptions) -> Result<Self> {
        let dataset = open_existing(uri, &options.storage_options)
            .await?
            .ok_or_else(|| Error::TableNotFound(uri.to_string()))?;
        check_format(&dataset)?;
        let defaults = table_defaults(&dataset, options)?;
        let stats = table_has_stats(&dataset);
        let schema = (Schema::from(dataset.schema()), dataset.schema().clone());
        let index = name_index(&schema.0);
        Ok(Self {
            uri: uri.to_string(),
            start_version: Some(dataset.version().version),
            detached: false,
            base: Some(dataset),
            overwrite: false,
            defaults,
            stats,
            options: options.clone(),
            schema: Some(schema),
            index,
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
            // One batch per run of rows with the same variables: a batch of
            // the union of all variables costs a column for each variable.
            let batches = runs(&group)
                .into_iter()
                .map(encode_rows)
                .collect::<Result<Vec<_>>>()?;
            for batch in &batches {
                self.promote(batch).await?;
            }
            let batches = batches
                .into_iter()
                .map(|batch| {
                    let table = self.staged_view(&batch.schema());
                    prepare_batch(
                        batch,
                        &self.defaults,
                        &self.options,
                        self.stats,
                        table.as_ref(),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            self.push(batches).await?;
        }
        Ok(())
    }

    /// The staged fields that `schema` names, or `None` before the first
    /// row. A lookup by name in the full schema takes a time linear in its
    /// fields.
    fn staged_view(&self, schema: &Schema) -> Option<Schema> {
        let (arrow, _) = self.schema.as_ref()?;
        let fields: Vec<FieldRef> = schema
            .fields()
            .iter()
            .filter_map(|field| self.index.get(field.name()))
            .map(|&position| arrow.fields()[position].clone())
            .collect();
        Some(Schema::new_with_metadata(fields, arrow.metadata().clone()))
    }

    /// Set the staged schema and its index.
    fn set_schema(&mut self, arrow: Schema, lance: LanceSchema) {
        self.index = name_index(&arrow);
        self.schema = Some((arrow, lance));
    }

    /// Write the rows of an encoded record batch, without a commit.
    ///
    /// `batch` holds nd columns and an optional `__global_attrs__` column,
    /// as [`crate::encode_rows`] gives them. The writer checks the
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
        if let Some(field) = batch.schema().fields().iter().find(|field| {
            is_stats_name(field.name())
                || (field.name() == GLOBAL_ATTRIBUTES_FIELD && !is_global_attributes_field(field))
        }) {
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
        let batch = nullable_global_attributes(batch)?;
        self.promote(&batch).await?;
        let table = self.staged_view(&batch.schema());
        let batch = prepare_batch(
            batch,
            &self.defaults,
            &self.options,
            self.stats,
            table.as_ref(),
        )?;
        self.push(vec![batch]).await
    }

    /// Widen the variables of `batch` in the table as [`Casting::Promote`]
    /// needs. The staged fragments use the old field ids that a widen
    /// replaces, so the rows written so far are committed first.
    async fn promote(&mut self, batch: &RecordBatch) -> Result<()> {
        if self.options.casting != Casting::Promote {
            return Ok(());
        }
        let Some(table) = self.staged_view(&batch.schema()) else {
            return Ok(());
        };
        let widenings = promotions(batch, &table);
        if widenings.is_empty() {
            return Ok(());
        }
        let result = async {
            if self.options.defer_promotion && self.start_version.is_some() && !self.detached {
                tracing::warn!(
                    uri = %self.uri,
                    "defer_promotion: no other writer may write to the table until finish"
                );
                self.detached = true;
            }
            let staged =
                !(self.buffer.is_empty() && self.pending.is_empty() && self.fragments.is_empty());
            if staged || self.base.is_none() {
                self.commit().await?;
            }
            let dataset = self.base.as_mut().expect("a committed table");
            widen_all(dataset, &widenings, self.detached)
                .boxed()
                .await?;
            let schema = (Schema::from(dataset.schema()), dataset.schema().clone());
            self.set_schema(schema.0, schema.1);
            Ok(())
        }
        .await;
        self.fail_on(result)
    }

    /// Grow the staged schema with `batches`, then buffer them. A full
    /// buffer becomes fragments.
    ///
    /// A schema conflict changes nothing, so it only rejects `batches`. A
    /// failed fragment write makes the writer fail.
    async fn push(&mut self, batches: Vec<RecordBatch>) -> Result<()> {
        self.grow(&batches)?;
        for batch in batches {
            self.buffered_bytes += batch.get_array_memory_size();
            self.buffer.push(batch);
            if self.written.on_disk(self.buffered_bytes)
                >= self.options.target_fragment_bytes as f64
            {
                let result = self.flush().boxed().await;
                self.fail_on(result)?;
            }
        }
        Ok(())
    }

    /// Add the fields of `batches` that the staged schema lacks. The time is
    /// linear in the fields of `batches`, plus one copy of the field list
    /// when the schema grows. A conflict changes nothing.
    fn grow(&mut self, batches: &[RecordBatch]) -> Result<()> {
        if self.schema.is_none() {
            let Some((first, rest)) = batches.split_first() else {
                return Ok(());
            };
            let arrow = first.schema().as_ref().clone();
            let lance = LanceSchema::try_from(&arrow)?;
            self.set_schema(arrow, lance);
            return self.grow(rest);
        }
        let (arrow, lance) = self.schema.as_ref().expect("a staged schema");
        // New top-level fields, and merged versions of staged fields.
        let mut added: Vec<FieldRef> = Vec::new();
        let mut added_at: HashMap<String, usize> = HashMap::new();
        let mut changed: HashMap<usize, FieldRef> = HashMap::new();
        for batch in batches {
            for field in batch.schema().fields() {
                let name = field.name();
                if let Some(&at) = added_at.get(name) {
                    if !holds(added[at].data_type(), field.data_type()) {
                        added[at] = Arc::new(merge_field(&added[at], field, name)?);
                    }
                } else if let Some(&position) = self.index.get(name) {
                    let current = changed.get(&position).unwrap_or(&arrow.fields()[position]);
                    if !holds(current.data_type(), field.data_type()) {
                        let merged = Arc::new(merge_field(current, field, name)?);
                        changed.insert(position, merged);
                    }
                } else {
                    added_at.insert(name.clone(), added.len());
                    added.push(field.clone());
                }
            }
        }
        if added.is_empty() && changed.is_empty() {
            return Ok(());
        }
        // All fallible steps first, so that a conflict changes nothing.
        // A merge leaves the new fields without ids, which set_field_id assigns.
        let added_lance = LanceSchema {
            fields: Vec::new(),
            metadata: Default::default(),
        }
        .merge(&Schema::new(added.clone()))?;
        let changed_lance = changed
            .iter()
            .map(|(&position, field)| {
                let current = LanceSchema {
                    fields: vec![lance.fields[position].clone()],
                    metadata: Default::default(),
                };
                let merged = current.merge(&Schema::new(vec![field.clone()]))?;
                let field = merged.fields.into_iter().next().expect("one field");
                Ok((position, field))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut fields: Vec<FieldRef> = arrow.fields().iter().cloned().collect();
        for (&position, field) in &changed {
            fields[position] = field.clone();
        }
        let first_added = fields.len();
        fields.extend(added.iter().cloned());
        let arrow = Schema::new_with_metadata(fields, arrow.metadata().clone());
        let max_field_id = self.base_max_field_id();
        let (_, mut lance) = self.schema.take().expect("a staged schema");
        for (position, field) in changed_lance {
            lance.fields[position] = field;
        }
        lance.fields.extend(added_lance.fields);
        lance.set_field_id(max_field_id);
        for (offset, field) in added.iter().enumerate() {
            self.index
                .insert(field.name().clone(), first_added + offset);
        }
        self.schema = Some((arrow, lance));
        Ok(())
    }

    /// Start a fragment write of the buffer. Wait for the oldest write when
    /// too many run.
    #[tracing::instrument(level = "debug", skip_all, fields(rows = self.buffer.len(), bytes = self.buffered_bytes))]
    async fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        self.evolve().await?;
        let batches = std::mem::take(&mut self.buffer);
        let memory = std::mem::take(&mut self.buffered_bytes);
        // One fragment per cluster of similar rows, each with only the
        // columns of its rows: the other columns read as null.
        let mut writes = Vec::new();
        for cluster in clusters(batches) {
            let positions = self.positions(&cluster)?;
            let (arrow, lance) = self.schema.as_ref().expect("a buffered row has a schema");
            let fields: Vec<FieldRef> = positions
                .iter()
                .map(|&position| arrow.fields()[position].clone())
                .collect();
            let present = Arc::new(Schema::new_with_metadata(fields, arrow.metadata().clone()));
            // A struct keeps all its children: Lance 11 cannot read a
            // fragment whose struct has fewer children than the table.
            let projected = LanceSchema {
                fields: positions
                    .iter()
                    .map(|&position| lance.fields[position].clone())
                    .collect(),
                metadata: lance.metadata.clone(),
            };
            // One batch per fragment: Lance 11 fails to read back a fragment of
            // several batches when a struct column has nulls and large lists.
            let batch = concat_conformed(&present, &cluster)?;
            writes.push((batch, present, projected));
        }
        let uri = self.uri.clone();
        let store_params = store_params(&self.options.storage_options);
        let parallelism = self.options.write_parallelism.max(1);
        let task = tokio::spawn(async move {
            let params = WriteParams {
                mode: LanceWriteMode::Append,
                data_storage_version: Some(FILE_VERSION),
                store_params,
                ..Default::default()
            };
            let written: Vec<Vec<Fragment>> = stream::iter(writes)
                .map(|(batch, present, projected)| {
                    let (uri, params) = (uri.clone(), params.clone());
                    async move {
                        let reader = RecordBatchIterator::new([Ok(batch)], present);
                        FragmentCreateBuilder::new(&uri)
                            .schema(&projected)
                            .write_params(&params)
                            .write_fragments(reader)
                            .await
                    }
                })
                .buffered(parallelism)
                .try_collect()
                .await?;
            Ok(written.into_iter().flatten().collect())
        });
        self.pending.push_back((task, memory));
        while self.pending.len() > self.options.write_parallelism.max(1) {
            self.wait_oldest().await?;
        }
        Ok(())
    }

    /// The positions in the staged schema of the columns of `batches`, in
    /// the order of the staged schema.
    fn positions(&self, batches: &[RecordBatch]) -> Result<Vec<usize>> {
        let mut positions = HashSet::new();
        for batch in batches {
            for field in batch.schema().fields() {
                let position = self.index.get(field.name()).ok_or_else(|| {
                    Error::InvalidTable(format!("no staged field '{}'", field.name()))
                })?;
                positions.insert(*position);
            }
        }
        let mut positions: Vec<usize> = positions.into_iter().collect();
        positions.sort_unstable();
        Ok(positions)
    }

    /// Add the staged fields that the table lacks to the table, and take the
    /// field ids of the table. Fragments then always use the ids of the table.
    ///
    /// A Project commit adds the fields: Lance validates a Merge commit, also
    /// that of `add_columns`, in a time that grows with the square of the
    /// schema fields.
    async fn evolve(&mut self) -> Result<()> {
        let (Some(base), Some((_, lance))) = (self.base.as_mut(), self.schema.as_ref()) else {
            return Ok(());
        };
        // A commit makes a table version, which a detached write must not
        // do. The final commit then adds the new fields.
        if self.overwrite || self.detached {
            return Ok(());
        }
        // New fields, also new struct children, get ids above those of the table.
        if lance.max_field_id() <= base.schema().max_field_id() {
            return Ok(());
        }
        commit_schema(base, lance.clone(), false).boxed().await?;
        let schema = (Schema::from(base.schema()), base.schema().clone());
        self.set_schema(schema.0, schema.1);
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
    #[tracing::instrument(skip_all, fields(uri = %self.uri), err)]
    pub async fn finish(mut self) -> Result<Dataset> {
        if let Some(error) = &self.failed {
            return Err(Error::WriterFailed(error.clone()));
        }
        // Boxed without its type: the commit future is large, and its full
        // type would exceed the compiler's layout and Send depth for callers.
        let dataset = self.commit().boxed().await?;
        match (self.detached, self.start_version) {
            (true, Some(start)) => publish(dataset, start).boxed().await,
            _ => Ok(dataset),
        }
    }

    /// Commit the staged rows. The writer then appends to the new version.
    #[tracing::instrument(level = "debug", skip_all, fields(uri = %self.uri))]
    async fn commit(&mut self) -> Result<Dataset> {
        self.flush().boxed().await?;
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
                    .with_detached(self.detached)
                    .execute(Transaction::new(version, operation, None))
                    .await?
            }
            // No new fields: new rows only, which Lance merges with concurrent appends.
            (Some(base), false) if schema.max_field_id() == base.schema().max_field_id() => {
                commit_rows(base, fragments, &schema, self.detached)
                    .boxed()
                    .await?
            }
            // New fields of a detached write: commit the schema, then the rows.
            (Some(mut base), false) => {
                commit_schema(&mut base, schema.clone(), self.detached)
                    .boxed()
                    .await?;
                commit_rows(base, fragments, &schema, self.detached)
                    .boxed()
                    .await?
            }
        };
        tracing::info!(version = dataset.version().version, "committed the rows");
        self.base = Some(dataset.clone());
        self.overwrite = false;
        self.set_schema(Schema::from(dataset.schema()), dataset.schema().clone());
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

/// The position of each top-level field of `schema`, by name.
fn name_index(schema: &Schema) -> HashMap<String, usize> {
    schema
        .fields()
        .iter()
        .enumerate()
        .map(|(position, field)| (field.name().clone(), position))
        .collect()
}

/// `rows` split into runs of consecutive rows with the same variables, in
/// the order of the rows.
fn runs(rows: &[NdRecordBatch]) -> Vec<&[NdRecordBatch]> {
    let mut runs = Vec::new();
    let mut start = 0;
    for end in 1..=rows.len() {
        if end == rows.len() || rows[end].schema() != rows[start].schema() {
            runs.push(&rows[start..end]);
            start = end;
        }
    }
    runs
}

/// The most columns of a fragment, as a multiple of its widest batch. A
/// fragment then holds at most this many absent values per value.
const CLUSTER_WIDTH: usize = 4;

/// The columns that a fragment can always hold, also with narrow batches.
const CLUSTER_MIN_COLUMNS: usize = 64;

/// `batches` split into runs of consecutive batches whose union of columns
/// stays at most [`CLUSTER_WIDTH`] times their widest batch, in order.
fn clusters(batches: Vec<RecordBatch>) -> Vec<Vec<RecordBatch>> {
    let mut clusters: Vec<Vec<RecordBatch>> = Vec::new();
    let mut names: HashSet<String> = HashSet::new();
    let mut widest = 0;
    for batch in batches {
        let schema = batch.schema();
        let new = schema
            .fields()
            .iter()
            .filter(|field| !names.contains(field.name()))
            .count();
        let limit = (CLUSTER_WIDTH * widest.max(batch.num_columns())).max(CLUSTER_MIN_COLUMNS);
        if clusters.is_empty() || names.len() + new > limit {
            clusters.push(Vec::new());
            names.clear();
            widest = 0;
        }
        names.extend(schema.fields().iter().map(|field| field.name().clone()));
        widest = widest.max(batch.num_columns());
        clusters.last_mut().expect("a cluster").push(batch);
    }
    clusters
}

/// `batch` with a nullable `__global_attrs__` field, as [`crate::encode_rows`]
/// gives it. Lance fills new attributes of older fragments with nulls, and
/// a read fails when the field is not nullable.
fn nullable_global_attributes(batch: RecordBatch) -> Result<RecordBatch> {
    let schema = batch.schema();
    let Some((index, field)) = schema.column_with_name(GLOBAL_ATTRIBUTES_FIELD) else {
        return Ok(batch);
    };
    if field.is_nullable() {
        return Ok(batch);
    }
    let mut fields: Vec<FieldRef> = schema.fields().iter().cloned().collect();
    fields[index] = Arc::new(field.clone().with_nullable(true));
    let schema = Schema::new_with_metadata(fields, schema.metadata().clone());
    Ok(RecordBatch::try_new(
        Arc::new(schema),
        batch.columns().to_vec(),
    )?)
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
/// the table still has each field of the fragments, and tries again. A
/// `detached` commit makes a detached version, which has no concurrent commits.
#[tracing::instrument(level = "debug", skip_all, fields(fragments = fragments.len()))]
async fn commit_rows(
    mut base: Dataset,
    fragments: Vec<Fragment>,
    staged: &LanceSchema,
    detached: bool,
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
            .with_detached(detached)
            .execute(Transaction::new(version, operation, None))
            .await
        {
            Ok(dataset) => return Ok(dataset),
            Err(lance::Error::RetryableCommitConflict { .. })
                if attempt < MAX_COMMIT_ATTEMPTS && !detached =>
            {
                tracing::debug!(attempt, "a concurrent commit preempted the rows, try again");
                base.checkout_latest().await?;
            }
            Err(error) => return Err(error.into()),
        }
    }
    unreachable!("the last attempt returns")
}

/// Make the detached version `staged` the table version, in one Restore
/// commit. Fails and commits nothing when the table changed after
/// `start_version`.
#[tracing::instrument(level = "debug", skip_all, fields(uri = %staged.uri()), err)]
async fn publish(mut staged: Dataset, start_version: u64) -> Result<Dataset> {
    let latest = staged.latest_version_id().await?;
    if latest != start_version {
        return Err(Error::CommitConflict {
            retryable: false,
            source: Box::new(lance::Error::invalid_input(format!(
                "with defer_promotion, the table must stay at version {start_version} \
                 until finish, but another writer committed version {latest}, \
                 nothing is committed"
            ))),
        });
    }
    // A Restore does not conflict with a concurrent commit, it replaces it.
    staged.restore().await?;
    tracing::info!(
        version = staged.version().version,
        "committed the rows and the widens"
    );
    Ok(staged)
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
