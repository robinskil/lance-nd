//! Read a Lance table as a stream of nd record batches.

use arrow::datatypes::{Field, Schema};
use futures::stream::{self, BoxStream};
use futures::{StreamExt, TryStreamExt};
use lance::Dataset;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use nd_arrow_array::NdRecordBatch;
use nd_arrow_array::extension::GLOBAL_ATTRIBUTES_FIELD;

use crate::encode::{RowDecoder, variable_field};
use crate::error::{Error, Result};

/// The variables of the table, as logical fields typed by their values.
pub fn variables(dataset: &Dataset) -> Result<Vec<Field>> {
    Schema::from(dataset.schema())
        .fields()
        .iter()
        .filter(|f| f.name() != GLOBAL_ATTRIBUTES_FIELD)
        .map(|f| variable_field(f))
        .collect()
}

/// A scan over the rows of the table.
pub fn scan(dataset: &Dataset) -> NdScanner<'_> {
    NdScanner {
        dataset,
        projection: None,
        filter: None,
        limit: None,
        batch_size: DEFAULT_BATCH_SIZE,
        batch_size_bytes: DEFAULT_BATCH_SIZE_BYTES,
        batch_readahead: None,
        decode_parallelism: None,
    }
}

/// Default rows in one Lance batch.
const DEFAULT_BATCH_SIZE: usize = 256;

/// Default bytes in one Lance batch. A large row gets a batch of its own.
const DEFAULT_BATCH_SIZE_BYTES: u64 = 32 << 20;

/// A scan that yields one nd record batch per row, in table order.
///
/// Lance reads batches of rows, limited by rows and by bytes. Several
/// batches decode to nd record batches at the same time.
pub struct NdScanner<'a> {
    dataset: &'a Dataset,
    projection: Option<Vec<String>>,
    filter: Option<String>,
    limit: Option<i64>,
    batch_size: usize,
    batch_size_bytes: u64,
    batch_readahead: Option<usize>,
    decode_parallelism: Option<usize>,
}

impl NdScanner<'_> {
    /// Read only these variables. A row keeps its global attributes.
    /// A row without any of these variables gives a batch without columns.
    pub fn project(mut self, variables: &[&str]) -> Self {
        self.projection = Some(variables.iter().map(|v| v.to_string()).collect());
        self
    }

    /// Read only the rows that match a Lance SQL filter.
    pub fn filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    /// Read at most `limit` rows.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit as i64);
        self
    }

    /// The most rows in one Lance batch. The default is 256.
    pub fn batch_size(mut self, rows: usize) -> Self {
        self.batch_size = rows.max(1);
        self
    }

    /// The approximate most bytes in one Lance batch. The default is 32 MiB.
    /// A row larger than this is a batch of its own.
    pub fn batch_size_bytes(mut self, bytes: u64) -> Self {
        self.batch_size_bytes = bytes.max(1);
        self
    }

    /// The most Lance batches that decode at the same time. The default is
    /// the number of compute CPUs of Lance.
    pub fn decode_parallelism(mut self, batches: usize) -> Self {
        self.decode_parallelism = Some(batches.max(1));
        self
    }

    /// The most Lance batches that Lance decodes ahead of the consumer.
    pub fn batch_readahead(mut self, batches: usize) -> Self {
        self.batch_readahead = Some(batches);
        self
    }

    /// A stream of the selected rows, in table order.
    pub async fn try_into_stream(self) -> Result<BoxStream<'static, Result<NdRecordBatch>>> {
        let mut scanner = self.dataset.scan();
        scanner.batch_size(self.batch_size);
        scanner.batch_size_bytes(self.batch_size_bytes);
        if let Some(projection) = &self.projection {
            let schema = self.dataset.schema();
            if let Some(unknown) = projection
                .iter()
                .find(|v| *v == GLOBAL_ATTRIBUTES_FIELD || schema.field(v).is_none())
            {
                return Err(Error::InvalidArgument(format!(
                    "'{unknown}' is not a variable of the table"
                )));
            }
            let mut columns = projection.clone();
            if schema.field(GLOBAL_ATTRIBUTES_FIELD).is_some() {
                columns.push(GLOBAL_ATTRIBUTES_FIELD.to_string());
            }
            scanner.project(&columns)?;
        }
        if let Some(filter) = &self.filter {
            scanner.filter(filter)?;
        }
        if self.limit.is_some() {
            scanner.limit(self.limit, None)?;
        }
        if let Some(rows) = self.batch_readahead {
            scanner.batch_readahead(rows);
        }
        let parallelism = self
            .decode_parallelism
            .unwrap_or_else(get_num_compute_intensive_cpus);
        let batches = scanner.try_into_stream().await?;
        // Decode on the Lance CPU pool. `buffered` keeps the table order.
        Ok(batches
            .map(|batch| async move {
                let batch = batch?;
                spawn_cpu(move || {
                    RowDecoder::try_new(&batch)?
                        .into_rows()
                        .collect::<Result<Vec<_>>>()
                })
                .await
            })
            .buffered(parallelism)
            .map_ok(|rows| stream::iter(rows.into_iter().map(Ok)))
            .try_flatten()
            .boxed())
    }

    /// Collect the selected rows.
    pub async fn try_collect(self) -> Result<Vec<NdRecordBatch>> {
        self.try_into_stream().await?.try_collect().await
    }
}
