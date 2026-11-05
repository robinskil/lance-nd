//! N-dimensional Arrow arrays in Lance.
//!
//! A table is a Lance dataset whose schema is the nd-arrow encoding: one
//! `nd_arrow.array` column per array name and an optional `__global_attrs__`
//! column. Each row is one [`NdRecordBatch`](nd_arrow_array::NdRecordBatch),
//! a complete one or a chunk of a larger one.
//!
//! - [`open()`]: open a table and check its format version.
//! - [`store`]: storage options for S3, Google Cloud Storage and Azure.
//! - [`versions()`], [`checkout()`] and [`restore()`]: read and restore earlier versions.
//! - [`write()`] and [`append()`]: write a stream of nd record batches.
//! - [`NdWriter`]: write in several calls and commit once.
//! - [`scan()`]: read a stream of nd record batches, with a projection.
//! - [`delete()`], [`compact()`] and [`cleanup_old_versions()`]: delete rows and
//!   reclaim space, see [`maintenance`].
//! - [`prune`]: change a DataFusion filter on values and attributes into a filter on the stats.
//! - [`stats`]: opt-in min and max values per variable and row, for pruning.
//! - [`ColumnOptions`]: Lance encoding options, like the compression, per column.
//!
//! See `FORMAT.md` for the table layout.

#![warn(missing_docs)]

pub mod cast;
pub mod encode;
pub mod error;
pub mod format;
pub mod maintenance;
pub mod options;
pub mod prune;
pub mod scan;
pub mod schema;
pub mod stats;
pub mod store;
pub mod versions;
pub mod widen;
pub mod write;
pub mod writer;

pub use cast::Casting;
pub use error::{Error, Result};
pub use format::{FORMAT_VERSION, check_format, open, open_with};
pub use lance::dataset::WriteMode;
pub use lance::dataset::cleanup::RemovalStats;
pub use lance::dataset::optimize::CompactionMetrics;
pub use lance::deps::datafusion;
pub use maintenance::{CompactOptions, cleanup_old_versions, compact, delete, delete_expr};
pub use options::{
    ByteStreamSplit, ColumnOptions, Compression, StructuralEncoding, column_defaults,
    set_column_defaults,
};
pub use scan::{NdScanner, scan, variables};
pub use store::StorageOptions;
pub use versions::{TableVersion, checkout, restore, versions};
pub use widen::widen;
pub use write::{WriteOptions, append, write};
pub use writer::NdWriter;
