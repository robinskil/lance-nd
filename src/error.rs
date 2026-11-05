//! Error type for lance-nd.

use std::sync::Arc;

use arrow::error::ArrowError;
use nd_arrow_array::NdError;

/// Errors raised by lance-nd.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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
    /// The Lance dataset does not follow the nd-arrow encoding, or has
    /// another format version.
    #[error("invalid table: {0}")]
    InvalidTable(String),
    /// An argument is not valid.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// No table exists at the uri.
    #[error("no table at {0}")]
    TableNotFound(String),
    /// A table exists at the uri.
    #[error("a table exists at {0}")]
    TableExists(String),
    /// Another commit changed the table first. When `retryable` is `true`,
    /// the same operation can succeed when you try it again.
    #[error("commit conflict: {source}")]
    CommitConflict {
        /// `true` when a new try can succeed.
        retryable: bool,
        /// The Lance error.
        source: Box<lance::Error>,
    },
    /// A write of an [`crate::NdWriter`] failed, so the writer commits nothing.
    #[error("the writer failed, nothing is committed: {0}")]
    WriterFailed(#[source] Arc<Error>),
    /// A fragment write task stopped without a result.
    #[error("a fragment write stopped: {0}")]
    TaskFailed(#[from] tokio::task::JoinError),
}

impl From<lance::Error> for Error {
    fn from(error: lance::Error) -> Self {
        match error {
            lance::Error::DatasetNotFound { path, .. } => Self::TableNotFound(path),
            lance::Error::DatasetAlreadyExists { uri, .. } => Self::TableExists(uri),
            lance::Error::CommitConflict { .. }
            | lance::Error::RetryableCommitConflict { .. }
            | lance::Error::TooMuchWriteContention { .. } => Self::CommitConflict {
                retryable: true,
                source: Box::new(error),
            },
            lance::Error::IncompatibleTransaction { .. } => Self::CommitConflict {
                retryable: false,
                source: Box::new(error),
            },
            error => Self::Lance(Box::new(error)),
        }
    }
}

/// A result with [`Error`] as the default error.
pub type Result<T, E = Error> = std::result::Result<T, E>;
