//! Read a Lance table as a stream of nd record batches.

use arrow::datatypes::{Field, Schema};
use futures::stream::{self, BoxStream};
use futures::{StreamExt, TryStreamExt};
use lance::Dataset;
use lance::dataset::scanner::{MaterializationStyle, Scanner};
use lance::deps::datafusion::logical_expr::Expr;
use lance::table::format::Fragment;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use nd_arrow_array::NdRecordBatch;
use nd_arrow_array::extension::GLOBAL_ATTRIBUTES_FIELD;

use crate::encode::{RowDecoder, variable_field};
use crate::error::{Error, Result};
use crate::prune::prune_filter;
use crate::stats::STATS_FIELD;

/// The variables of the table, as logical fields typed by their values.
pub fn variables(dataset: &Dataset) -> Result<Vec<Field>> {
    Schema::from(dataset.schema())
        .fields()
        .iter()
        .filter(|f| f.name() != GLOBAL_ATTRIBUTES_FIELD && f.name() != STATS_FIELD)
        .map(|f| variable_field(f))
        .collect()
}

/// A scan over the rows of the table.
pub fn scan(dataset: &Dataset) -> NdScanner<'_> {
    NdScanner {
        dataset,
        projection: None,
        filter: None,
        filter_expr: None,
        lance_filter: None,
        fragments: None,
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
    filter_expr: Option<Expr>,
    lance_filter: Option<Expr>,
    fragments: Option<Vec<Fragment>>,
    limit: Option<i64>,
    batch_size: usize,
    batch_size_bytes: u64,
    batch_readahead: Option<usize>,
    decode_parallelism: Option<usize>,
}

impl NdScanner<'_> {
    /// Read only these variables. A row keeps its global attributes.
    /// A row without any of these variables gives a batch without columns,
    /// so an empty list reads only the global attributes.
    ///
    /// Each name must be a whole variable. A nested field, such as
    /// `sst.attrs.units`, gives [`Error::InvalidArgument`] in the scan.
    pub fn project(mut self, variables: &[&str]) -> Self {
        self.projection = Some(variables.iter().map(|v| v.to_string()).collect());
        self
    }

    /// Read only the rows that match a Lance SQL filter.
    ///
    /// A scan cannot have both this filter and a DataFusion filter, see
    /// [`NdScanner::filter_expr`] and [`NdScanner::lance_filter`].
    pub fn filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    /// Skip the rows that surely have no value that matches `expr`.
    ///
    /// `expr` is a DataFusion filter on the table columns, for example
    /// `get_field(col("lat"), "values").between(lit(17.5), lit(25.4))`. A
    /// part on `<variable>.values` uses the stats of the table, see
    /// [`crate::prune`]. A kept row comes back whole, also with values
    /// outside the filter. A table without stats keeps all rows for a value
    /// part. Each other part, for example on attributes, filters the rows
    /// exactly.
    pub fn filter_expr(mut self, expr: Expr) -> Self {
        self.filter_expr = Some(expr);
        self
    }

    /// Read only the rows that match `expr`, a DataFusion filter on the
    /// Lance table columns.
    ///
    /// Lance gets `expr` with no change, so `expr` must use the table
    /// columns: the nd columns, `__global_attrs__` and `__stats__`. With
    /// [`NdScanner::filter_expr`], a row must match both filters.
    pub fn lance_filter(mut self, expr: Expr) -> Self {
        self.lance_filter = Some(expr);
        self
    }

    /// Read only the rows of these fragments, in the order given.
    pub fn fragments(mut self, fragments: Vec<Fragment>) -> Self {
        self.fragments = Some(fragments);
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

    /// The Lance plan of the scan, for example to see which columns the
    /// filter read loads.
    pub async fn explain_plan(&self, verbose: bool) -> Result<String> {
        let (scanner, _) = self.lance_scanner()?;
        Ok(scanner.explain_plan(verbose).await?)
    }

    /// A stream of the selected rows, in table order.
    pub async fn try_into_stream(self) -> Result<BoxStream<'static, Result<NdRecordBatch>>> {
        let (scanner, rows_only) = self.lance_scanner()?;
        let parallelism = self
            .decode_parallelism
            .unwrap_or_else(get_num_compute_intensive_cpus);
        let batches = scanner.try_into_stream().await?;
        // Decode on the Lance CPU pool. `buffered` keeps the table order.
        Ok(batches
            .map(move |batch| async move {
                let batch = match rows_only {
                    true => batch?.project(&[])?,
                    false => batch?,
                };
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

    /// The Lance scanner, and `true` when it reads only row ids.
    fn lance_scanner(&self) -> Result<(Scanner, bool)> {
        let mut scanner = self.dataset.scan();
        scanner.batch_size(self.batch_size);
        scanner.batch_size_bytes(self.batch_size_bytes);
        let mut rows_only = false;
        if let Some(projection) = &self.projection {
            let schema = self.dataset.schema();
            let is_column = |name: &str| schema.fields.iter().any(|f| f.name == name);
            if let Some(unknown) = projection
                .iter()
                .find(|v| *v == GLOBAL_ATTRIBUTES_FIELD || *v == STATS_FIELD || !is_column(v))
            {
                return Err(Error::InvalidArgument(format!(
                    "'{unknown}' is not a variable of the table: a projection takes whole variables, not nested fields"
                )));
            }
            let mut columns = projection.clone();
            if is_column(GLOBAL_ATTRIBUTES_FIELD) {
                columns.push(GLOBAL_ATTRIBUTES_FIELD.to_string());
            }
            // Lance needs one column, so read the row ids and drop them.
            rows_only = columns.is_empty();
            if rows_only {
                scanner.with_row_id();
            }
            scanner.project(&columns)?;
        }
        if self.filter.is_some() && (self.filter_expr.is_some() || self.lance_filter.is_some()) {
            return Err(Error::InvalidArgument(
                "a scan cannot have both filter and a DataFusion filter".to_string(),
            ));
        }
        let mut filtered = false;
        if let Some(filter) = &self.filter {
            scanner.filter(filter)?;
            filtered = true;
        }
        let pruned = match &self.filter_expr {
            Some(expr) => prune_filter(expr, &Schema::from(self.dataset.schema()))?,
            None => None,
        };
        if let Some(expr) = pruned
            .into_iter()
            .chain(self.lance_filter.clone())
            .reduce(Expr::and)
        {
            scanner.filter_expr(expr);
            filtered = true;
        }
        // Read the filter columns first, then the variables of the matching
        // rows only. Lance reads a struct of lists early by default.
        if filtered {
            scanner.materialization_style(MaterializationStyle::AllLate);
        }
        if let Some(fragments) = &self.fragments {
            scanner.with_fragments(fragments.clone());
        }
        if self.limit.is_some() {
            scanner.limit(self.limit, None)?;
        }
        if let Some(rows) = self.batch_readahead {
            scanner.batch_readahead(rows);
        }
        Ok((scanner, rows_only))
    }

    /// Collect the selected rows.
    pub async fn try_collect(self) -> Result<Vec<NdRecordBatch>> {
        self.try_into_stream().await?.try_collect().await
    }
}
