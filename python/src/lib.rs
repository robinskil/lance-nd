//! Python bindings for lance-nd.
//!
//! The module `lance_nd._lance_nd` holds the native classes. The Python
//! package `lance_nd` wraps them and adds the xarray encoder.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Schema};
use arrow::pyarrow::PyArrowType;
use futures::StreamExt;
use futures::stream::BoxStream;
use lance::Dataset;
use lance_datafusion::planner::Planner;
use lance_nd::datafusion::common::DFSchema;
use lance_nd::datafusion::prelude::{SessionConfig, SessionContext};
use lance_nd::encode::encode_rows;
use lance_nd::stats::has_stats;
use lance_nd::{
    ByteStreamSplit, Casting, ColumnOptions, CompactOptions, Compression, NdWriter,
    StructuralEncoding, WriteMode, WriteOptions, scan, variables,
};
use nd_arrow_array::NdRecordBatch;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::runtime::Runtime;

create_exception!(_lance_nd, LanceNdError, PyException);

fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| Runtime::new().expect("a Tokio runtime"))
}

fn to_py(error: lance_nd::Error) -> PyErr {
    LanceNdError::new_err(error.to_string())
}

fn closed() -> PyErr {
    LanceNdError::new_err("writer is closed")
}

/// Run `future` on the runtime without the GIL.
fn block_on<F: Future + Send>(py: Python<'_>, future: F) -> F::Output
where
    F::Output: Send,
{
    py.detach(|| runtime().block_on(future))
}

/// A writer that makes all its datasets visible in one commit at `finish()`.
#[pyclass(module = "lance_nd._lance_nd")]
struct Writer {
    inner: Mutex<Option<NdWriter>>,
    uri: String,
}

impl Writer {
    fn take(&self) -> PyResult<NdWriter> {
        self.inner.lock().unwrap().take().ok_or_else(closed)
    }
}

#[pymethods]
impl Writer {
    /// A writer for a new table. `mode` is "create" or "overwrite".
    /// `stats` stores the min and max of each variable per row. `casting`
    /// is "no", "safe", "same_kind" or "promote".
    #[staticmethod]
    #[pyo3(signature = (
        uri,
        mode = "create",
        column_defaults = None,
        column_options = None,
        stats = false,
        target_fragment_bytes = None,
        write_parallelism = None,
        casting = "no",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn create(
        py: Python<'_>,
        uri: String,
        mode: &str,
        column_defaults: Option<&Bound<'_, PyDict>>,
        column_options: Option<&Bound<'_, PyDict>>,
        stats: bool,
        target_fragment_bytes: Option<usize>,
        write_parallelism: Option<usize>,
        casting: &str,
    ) -> PyResult<Self> {
        let mode = match mode {
            "create" => WriteMode::Create,
            "overwrite" => WriteMode::Overwrite,
            other => {
                return Err(PyValueError::new_err(format!(
                    "mode must be 'create' or 'overwrite', not '{other}'"
                )));
            }
        };
        let options = fragment_options(
            WriteOptions {
                mode,
                column_defaults: column_defaults.map(parse_options).transpose()?,
                column_options: parse_column_options(column_options)?,
                stats,
                casting: parse_casting(casting)?,
                ..Default::default()
            },
            target_fragment_bytes,
            write_parallelism,
        );
        let writer = block_on(py, NdWriter::create(&uri, &options)).map_err(to_py)?;
        Ok(Self {
            inner: Mutex::new(Some(writer)),
            uri,
        })
    }

    /// A writer that appends to an existing table.
    #[staticmethod]
    #[pyo3(signature = (
        uri,
        column_options = None,
        target_fragment_bytes = None,
        write_parallelism = None,
        casting = "no",
    ))]
    fn append(
        py: Python<'_>,
        uri: String,
        column_options: Option<&Bound<'_, PyDict>>,
        target_fragment_bytes: Option<usize>,
        write_parallelism: Option<usize>,
        casting: &str,
    ) -> PyResult<Self> {
        let options = fragment_options(
            WriteOptions {
                column_options: parse_column_options(column_options)?,
                casting: parse_casting(casting)?,
                ..Default::default()
            },
            target_fragment_bytes,
            write_parallelism,
        );
        let writer = block_on(py, NdWriter::append(&uri, &options)).map_err(to_py)?;
        Ok(Self {
            inner: Mutex::new(Some(writer)),
            uri,
        })
    }

    /// Write the datasets of an nd-arrow record batch, one per row, without a commit.
    fn write_batch(&self, py: Python<'_>, batch: PyArrowType<RecordBatch>) -> PyResult<()> {
        let mut writer = self.take()?;
        let result = block_on(py, writer.write_batch(batch.0));
        // The writer stays open after a failed write; its finish() then fails.
        *self.inner.lock().unwrap() = Some(writer);
        result.map_err(to_py)
    }

    /// Commit all written datasets in one new table version.
    fn finish(&self, py: Python<'_>) -> PyResult<Table> {
        let writer = self.take()?;
        let dataset = block_on(py, writer.finish()).map_err(to_py)?;
        Ok(Table {
            dataset,
            uri: self.uri.clone(),
        })
    }

    /// Drop all written datasets without a commit.
    fn abort(&self) -> PyResult<()> {
        self.take()?.abort();
        Ok(())
    }
}

/// A lance-nd table at one version.
#[pyclass(module = "lance_nd._lance_nd")]
struct Table {
    dataset: Dataset,
    uri: String,
}

#[pymethods]
impl Table {
    /// Open the table at `uri`.
    #[staticmethod]
    fn open(py: Python<'_>, uri: String) -> PyResult<Self> {
        let dataset = block_on(py, Dataset::open(&uri))
            .map_err(|error| to_py(lance_nd::Error::from(error)))?;
        Ok(Self { dataset, uri })
    }

    /// The uri of the table.
    #[getter]
    fn uri(&self) -> &str {
        &self.uri
    }

    /// The Lance version of the table.
    #[getter]
    fn version(&self) -> u64 {
        self.dataset.version().version
    }

    /// `true` when the table stores the min and max of each variable per row.
    #[getter]
    fn has_stats(&self) -> bool {
        has_stats(&self.dataset.schema().into())
    }

    /// The number of Lance fragments of the table.
    #[getter]
    fn num_fragments(&self) -> usize {
        self.dataset.get_fragments().len()
    }

    /// The number of datasets in the table.
    fn count_rows(&self, py: Python<'_>) -> PyResult<usize> {
        block_on(py, self.dataset.count_rows(None))
            .map_err(|error| to_py(lance_nd::Error::from(error)))
    }

    /// The names of the variables in the table.
    fn variables(&self) -> PyResult<Vec<String>> {
        let fields = variables(&self.dataset).map_err(to_py)?;
        Ok(fields.iter().map(|f| f.name().clone()).collect())
    }

    /// Read the datasets of the table as a stream of nd record batches.
    ///
    /// `filter` is an exact Lance SQL filter. `prune` is a Lance SQL filter
    /// whose `<variable>.values` parts use the stats to skip datasets.
    #[pyo3(signature = (
        project = None,
        filter = None,
        prune = None,
        limit = None,
        batch_size = None,
        batch_size_bytes = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn scan(
        &self,
        py: Python<'_>,
        project: Option<Vec<String>>,
        filter: Option<String>,
        prune: Option<String>,
        limit: Option<usize>,
        batch_size: Option<usize>,
        batch_size_bytes: Option<u64>,
    ) -> PyResult<Scanner> {
        // Python has no DataFusion expressions, so the filters are SQL text.
        let schema = Arc::new(Schema::from(self.dataset.schema()));
        let planner = Planner::new(schema.clone());
        let parse = |sql: &str| {
            planner
                .parse_filter(sql)
                .map_err(|error| to_py(lance_nd::Error::from(error)))
        };
        // Lance coerces literals to the column type, and fails for a literal
        // on `<variable>.values`. DataFusion parses without the coercion, and
        // keeps the case of names as Lance does.
        let config = SessionConfig::new()
            .set_bool("datafusion.sql_parser.enable_ident_normalization", false);
        let context = SessionContext::new_with_config(config);
        let parse_values = |sql: &str| {
            let expr = DFSchema::try_from(schema.as_ref().clone())
                .and_then(|schema| context.parse_sql_expr(sql, &schema));
            expr.map_err(|error| to_py(lance_nd::Error::from(lance::Error::from(error))))
        };
        let mut scanner = scan(&self.dataset);
        if let Some(project) = &project {
            let names: Vec<&str> = project.iter().map(String::as_str).collect();
            scanner = scanner.project(&names);
        }
        if let Some(filter) = &filter {
            scanner = scanner.lance_filter(parse(filter)?);
        }
        if let Some(prune) = &prune {
            scanner = scanner.filter_expr(parse_values(prune)?);
        }
        if let Some(limit) = limit {
            scanner = scanner.limit(limit);
        }
        if let Some(rows) = batch_size {
            scanner = scanner.batch_size(rows);
        }
        if let Some(bytes) = batch_size_bytes {
            scanner = scanner.batch_size_bytes(bytes);
        }
        let rows = block_on(py, scanner.try_into_stream()).map_err(to_py)?;
        Ok(Scanner {
            rows: Mutex::new(Some(rows)),
        })
    }

    /// Change the value type of `variable` to the wider type `to`.
    fn widen(
        &mut self,
        py: Python<'_>,
        variable: String,
        to: PyArrowType<DataType>,
    ) -> PyResult<()> {
        block_on(py, lance_nd::widen(&mut self.dataset, &variable, &to.0)).map_err(to_py)
    }

    /// Delete the datasets that match `predicate`, a Lance SQL filter.
    /// Returns the number of deleted datasets.
    fn delete(&mut self, py: Python<'_>, predicate: String) -> PyResult<u64> {
        block_on(py, lance_nd::delete(&mut self.dataset, &predicate)).map_err(to_py)
    }

    /// Merge small fragments and remove deleted datasets.
    #[pyo3(signature = (target_fragment_bytes = None, deleted_fraction = None))]
    fn compact(
        &mut self,
        py: Python<'_>,
        target_fragment_bytes: Option<u64>,
        deleted_fraction: Option<f32>,
    ) -> PyResult<HashMap<&'static str, usize>> {
        let defaults = CompactOptions::default();
        let options = CompactOptions {
            target_fragment_bytes: target_fragment_bytes.unwrap_or(defaults.target_fragment_bytes),
            deleted_fraction: deleted_fraction.unwrap_or(defaults.deleted_fraction),
        };
        let metrics =
            block_on(py, lance_nd::compact(&mut self.dataset, &options)).map_err(to_py)?;
        Ok(HashMap::from([
            ("fragments_removed", metrics.fragments_removed),
            ("fragments_added", metrics.fragments_added),
            ("files_removed", metrics.files_removed),
            ("files_added", metrics.files_added),
        ]))
    }

    /// Remove the files of the table versions older than `older_than`.
    #[pyo3(signature = (older_than = Duration::from_secs(14 * 24 * 60 * 60)))]
    fn cleanup_old_versions(
        &self,
        py: Python<'_>,
        older_than: Duration,
    ) -> PyResult<HashMap<&'static str, u64>> {
        let stats = block_on(
            py,
            lance_nd::cleanup_old_versions(&self.dataset, older_than),
        )
        .map_err(to_py)?;
        Ok(HashMap::from([
            ("bytes_removed", stats.bytes_removed),
            ("old_versions", stats.old_versions),
        ]))
    }
}

/// The whole table as one nd-arrow record batch. For tests only.
/// A stream of nd record batches, one per dataset of the table.
#[pyclass(module = "lance_nd._lance_nd")]
struct Scanner {
    rows: Mutex<Option<BoxStream<'static, lance_nd::Result<NdRecordBatch>>>>,
}

#[pymethods]
impl Scanner {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// The next dataset as a record batch of one row.
    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyArrowType<RecordBatch>>> {
        let mut rows = self.rows.lock().map_err(|_| closed())?;
        let Some(stream) = rows.as_mut() else {
            return Ok(None);
        };
        match block_on(py, stream.next()) {
            Some(row) => {
                let row = row.map_err(to_py)?;
                Ok(Some(PyArrowType(encode_rows(&[row]).map_err(to_py)?)))
            }
            None => {
                *rows = None;
                Ok(None)
            }
        }
    }
}

#[pyfunction]
fn _scan_batch(py: Python<'_>, uri: String) -> PyResult<PyArrowType<RecordBatch>> {
    let batch = block_on(py, async {
        let dataset = Dataset::open(&uri).await?;
        let rows = scan(&dataset).try_collect().await?;
        encode_rows(&rows)
    })
    .map_err(to_py)?;
    Ok(PyArrowType(batch))
}

/// `options` with the fragment size and the write parallelism, when given.
fn parse_casting(casting: &str) -> PyResult<Casting> {
    match casting {
        "no" => Ok(Casting::No),
        "safe" => Ok(Casting::Safe),
        "same_kind" => Ok(Casting::SameKind),
        "promote" => Ok(Casting::Promote),
        other => Err(invalid(
            "casting",
            other,
            "'no', 'safe', 'same_kind' or 'promote'",
        )),
    }
}

fn fragment_options(
    mut options: WriteOptions,
    target_fragment_bytes: Option<usize>,
    write_parallelism: Option<usize>,
) -> WriteOptions {
    if let Some(bytes) = target_fragment_bytes {
        options.target_fragment_bytes = bytes;
    }
    if let Some(tasks) = write_parallelism {
        options.write_parallelism = tasks;
    }
    options
}

/// Encoding options for each named column.
fn parse_column_options(
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<HashMap<String, ColumnOptions>> {
    let Some(options) = options else {
        return Ok(HashMap::new());
    };
    options
        .iter()
        .map(|(name, value)| {
            let name: String = name.extract()?;
            let value = value.cast_into::<PyDict>().map_err(|_| {
                PyValueError::new_err(format!("the options of column '{name}' must be a dict"))
            })?;
            Ok((name, parse_options(&value)?))
        })
        .collect()
}

/// Encoding options of one column, with the keys of `ColumnOptions`.
fn parse_options(options: &Bound<'_, PyDict>) -> PyResult<ColumnOptions> {
    let mut parsed = ColumnOptions::default();
    let mut level: Option<i32> = None;
    for (key, value) in options.iter() {
        let key: String = key.extract()?;
        match key.as_str() {
            "compression" => {
                parsed.compression = Some(match value.extract::<String>()?.as_str() {
                    "none" => Compression::None,
                    "lz4" => Compression::Lz4,
                    "zstd" => Compression::Zstd { level: None },
                    other => return Err(invalid(&key, other, "'none', 'lz4' or 'zstd'")),
                });
            }
            "compression_level" => level = Some(value.extract()?),
            "byte_stream_split" => {
                parsed.byte_stream_split = Some(match value.extract::<String>()?.as_str() {
                    "off" => ByteStreamSplit::Off,
                    "on" => ByteStreamSplit::On,
                    "auto" => ByteStreamSplit::Auto,
                    other => return Err(invalid(&key, other, "'off', 'on' or 'auto'")),
                });
            }
            "rle_threshold" => parsed.rle_threshold = Some(value.extract()?),
            "structural_encoding" => {
                parsed.structural_encoding = Some(match value.extract::<String>()?.as_str() {
                    "miniblock" => StructuralEncoding::MiniBlock,
                    "fullzip" => StructuralEncoding::FullZip,
                    other => return Err(invalid(&key, other, "'miniblock' or 'fullzip'")),
                });
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown column option '{other}'"
                )));
            }
        }
    }
    if let Some(level) = level {
        match parsed.compression {
            Some(Compression::Zstd { .. }) => {
                parsed.compression = Some(Compression::Zstd { level: Some(level) });
            }
            _ => {
                return Err(PyValueError::new_err(
                    "compression_level needs compression 'zstd'",
                ));
            }
        }
    }
    Ok(parsed)
}

fn invalid(key: &str, value: &str, expected: &str) -> PyErr {
    PyValueError::new_err(format!("{key} must be {expected}, not '{value}'"))
}

#[pymodule]
fn _lance_nd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("LanceNdError", m.py().get_type::<LanceNdError>())?;
    m.add_class::<Writer>()?;
    m.add_class::<Table>()?;
    m.add_class::<Scanner>()?;
    m.add_function(wrap_pyfunction!(_scan_batch, m)?)?;
    Ok(())
}
