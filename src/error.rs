//! Error type for lance-nd.

use arrow::error::ArrowError;
use nd_arrow_array::NdError;

/// Errors raised by lance-nd.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An error from the nd array encoding.
    #[error(transparent)]
    Nd(#[from] NdError),
    /// An error from Lance.
    #[error(transparent)]
    Lance(Box<lance::Error>),
    /// An error from Arrow.
    #[error(transparent)]
    Arrow(#[from] ArrowError),
    /// Variable names or types do not match.
    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),
    /// The Lance dataset does not follow the nd-arrow encoding.
    #[error("invalid table: {0}")]
    InvalidTable(String),
    /// An argument is not valid.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

impl From<lance::Error> for Error {
    fn from(error: lance::Error) -> Self {
        Self::Lance(Box::new(error))
    }
}

/// A result with [`Error`] as the default error.
pub type Result<T, E = Error> = std::result::Result<T, E>;
