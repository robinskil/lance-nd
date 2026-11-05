//! N-dimensional Arrow arrays in Lance.
//!
//! A table is a Lance dataset whose schema is the nd-arrow encoding: one
//! `nd_arrow.array` column per array name and an optional `__global_attrs__`
//! column. Each row is one [`NdRecordBatch`](nd_arrow_array::NdRecordBatch),
//! a complete one or a chunk of a larger one.
//!
//! - [`open()`] and [`open_with()`]: open a table and check its format
//!   version. [`StorageOptions`] hold the credentials of S3, Google Cloud
//!   Storage or Azure.
//! - [`write()`] and [`append()`]: write a stream of nd record batches.
//! - [`NdWriter`]: write in several calls and commit once.
//! - [`scan()`]: read a stream of nd record batches, with a projection and
//!   filters. [`prune_filter()`] shows how a filter on values uses the stats.
//! - [`delete()`], [`compact()`] and [`cleanup_old_versions()`]: delete rows and
//!   reclaim space.
//! - [`versions()`], [`checkout()`] and [`restore()`]: read and restore earlier versions.
//! - [`widen()`]: change the value type of a variable.
//! - [`ColumnOptions`]: Lance encoding options, like the compression, per column.
//! - [`encode_rows()`] and [`RowDecoder`]: the row encoding, for record
//!   batches that you write or read yourself.
//!
//! See `FORMAT.md` for the table layout.

#![warn(missing_docs)]

mod cast;
mod encode;
mod error;
mod format;
mod maintenance;
mod options;
mod prune;
mod scan;
mod schema;
mod stats;
mod store;
mod versions;
mod widen;
mod write;
mod writer;

pub use cast::Casting;
pub use encode::{RowDecoder, encode_rows};
pub use error::{Error, Result};
pub use format::{FORMAT_VERSION, check_format, open, open_with};
/// The DataFusion version of Lance, to build filters.
pub use lance::deps::datafusion;
pub use maintenance::{
    CleanupStats, CompactOptions, CompactionStats, cleanup_old_versions, compact, delete,
    delete_expr,
};
pub use options::{
    ByteStreamSplit, ColumnOptions, Compression, StructuralEncoding, column_defaults,
    set_column_defaults,
};
pub use prune::prune_filter;
pub use scan::{NdScanner, scan, variables};
pub use stats::{STATS_PREFIX, has_stats, stats_column, stats_variable};
pub use store::StorageOptions;
pub use versions::{TableVersion, checkout, restore, versions};
pub use widen::widen;
pub use write::{WriteMode, WriteOptions, append, write};
pub use writer::NdWriter;
