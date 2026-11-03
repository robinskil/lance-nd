//! N-dimensional Arrow arrays in Lance.
//!
//! A table is a Lance dataset whose schema is the nd-arrow encoding: one
//! `nd_arrow.array` column per array name and an optional `__global_attrs__`
//! column. Each row is one [`NdRecordBatch`](nd_arrow_array::NdRecordBatch),
//! a complete one or a chunk of a larger one.
//!
//! - [`write()`] and [`append()`]: write a stream of nd record batches.
//! - [`NdWriter`]: write in several calls and commit once.
//! - [`scan()`]: read a stream of nd record batches, with a projection.
//! - [`prune`]: change a DataFusion filter on values and attributes into a filter on the stats.
//! - [`stats`]: opt-in min and max values per variable and row, for pruning.
//! - [`ColumnOptions`]: Lance encoding options, like the compression, per column.
//!
//! See `FORMAT.md` for the table layout.

#![warn(missing_docs)]

pub mod encode;
pub mod error;
pub mod options;
pub mod prune;
pub mod scan;
pub mod schema;
pub mod stats;
pub mod write;
pub mod writer;

pub use error::{Error, Result};
pub use lance::dataset::WriteMode;
pub use lance::deps::datafusion;
pub use options::{
    ByteStreamSplit, ColumnOptions, Compression, StructuralEncoding, column_defaults,
    set_column_defaults,
};
pub use scan::{NdScanner, scan, variables};
pub use write::{WriteOptions, append, write};
pub use writer::NdWriter;
