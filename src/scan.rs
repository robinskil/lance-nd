//! Read a Lance table as a stream of nd record batches.

use arrow::datatypes::{Field, Schema};
use futures::stream::{self, BoxStream};
use futures::{StreamExt, TryStreamExt};
use lance::Dataset;
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
        batch_readahead: None,
    }
}

/// A scan that yields one nd record batch per row.
///
/// The scan reads one row at a time from Lance, so a large row is not held
/// with its neighbours.
pub struct NdScanner<'a> {
    dataset: &'a Dataset,
    projection: Option<Vec<String>>,
    filter: Option<String>,
    limit: Option<i64>,
    batch_readahead: Option<usize>,
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

    /// The most rows that Lance decodes ahead of the consumer.
    pub fn batch_readahead(mut self, rows: usize) -> Self {
        self.batch_readahead = Some(rows);
        self
    }

    /// A stream of the selected rows, in table order.
    pub async fn try_into_stream(self) -> Result<BoxStream<'static, Result<NdRecordBatch>>> {
        let mut scanner = self.dataset.scan();
        // One row per Lance batch: each row is a whole nd record batch.
        scanner.batch_size(1);
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
        let batches = scanner.try_into_stream().await?;
        Ok(batches
            .map_err(Error::from)
            .and_then(|batch| async move { RowDecoder::try_new(&batch) })
            .map_ok(|decoder| stream::iter(decoder.into_rows()))
            .try_flatten()
            .boxed())
    }

    /// Collect the selected rows.
    pub async fn try_collect(self) -> Result<Vec<NdRecordBatch>> {
        self.try_into_stream().await?.try_collect().await
    }
}
