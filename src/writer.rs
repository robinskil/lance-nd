//! Write nd record batches in several calls and commit them once.

use std::pin::pin;
use std::sync::Arc;

use arrow::array::{RecordBatch, RecordBatchIterator};
use arrow::datatypes::Schema;
use futures::Stream;
use lance::Dataset;
use lance::dataset::fragment::write::FragmentCreateBuilder;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{CommitBuilder, WriteMode, WriteParams};
use lance::table::format::Fragment;
use lance_core::datatypes::{OnMissing, OnTypeMismatch, Schema as LanceSchema};
use nd_arrow_array::NdRecordBatch;

use crate::error::{Error, Result};
use crate::options::ColumnOptions;
use crate::schema::{conform_batch, merge_schemas};
use crate::write::{
    FILE_VERSION, WriteOptions, encode_group, next_group, table_defaults, table_has_stats,
};

/// A writer that makes all its rows visible in one commit.
///
/// [`NdWriter::write`] writes Lance data files, but readers do not see them.
/// [`NdWriter::finish`] commits all of them in one new table version.
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
    fragments: Vec<Fragment>,
    /// The error of a failed write. `finish` then fails too.
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
            fragments: Vec::new(),
            failed: None,
        })
    }

    /// Write a stream of rows as data files, without a commit.
    /// Each nd record batch is one row.
    ///
    /// A failed write makes the writer fail: [`NdWriter::finish`] then
    /// commits nothing.
    pub async fn write(&mut self, rows: impl Stream<Item = Result<NdRecordBatch>>) -> Result<()> {
        if let Some(error) = &self.failed {
            return Err(Error::InvalidArgument(format!(
                "an earlier write failed: {error}"
            )));
        }
        let result = self.write_rows(rows).await;
        if let Err(error) = &result {
            self.failed = Some(error.to_string());
        }
        result
    }

    async fn write_rows(&mut self, rows: impl Stream<Item = Result<NdRecordBatch>>) -> Result<()> {
        let mut rows = pin!(rows);
        while let Some(group) = next_group(&mut rows, &self.options).await? {
            let batch = encode_group(&group, &self.defaults, &self.options, self.stats)?;
            self.write_batch(batch).await?;
        }
        Ok(())
    }

    /// Grow the staged schema, then write `batch` with its field ids.
    async fn write_batch(&mut self, batch: RecordBatch) -> Result<()> {
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
        // Only the columns of the batch: the other fields read as null.
        let present: Vec<_> = arrow
            .fields()
            .iter()
            .filter(|field| batch.schema().column_with_name(field.name()).is_some())
            .cloned()
            .collect();
        let present = Arc::new(Schema::new_with_metadata(present, arrow.metadata().clone()));
        let batch = conform_batch(&batch, &present)?;
        let projected =
            lance.project_by_schema(present.as_ref(), OnMissing::Error, OnTypeMismatch::Error)?;
        let params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(FILE_VERSION),
            ..Default::default()
        };
        let reader = RecordBatchIterator::new([Ok(batch)], present);
        let fragments = FragmentCreateBuilder::new(&self.uri)
            .schema(&projected)
            .write_params(&params)
            .write_fragments(reader)
            .await?;
        self.fragments.extend(fragments);
        self.schema = Some((arrow, lance));
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
    pub async fn finish(self) -> Result<Dataset> {
        if let Some(error) = self.failed {
            return Err(Error::InvalidArgument(format!(
                "an earlier write failed, nothing is committed: {error}"
            )));
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

    /// Drop all written rows without a commit.
    pub fn abort(self) {}
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
