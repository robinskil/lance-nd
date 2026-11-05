//! Read and restore earlier versions of a table.
//!
//! Each commit makes a new table version. The files of a version stay until
//! [`crate::cleanup_old_versions`] removes them.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use lance::Dataset;

use crate::error::Result;
use crate::format::check_format;

/// One version of a table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableVersion {
    /// The version number. The first version is 1.
    pub version: u64,
    /// The time of the commit.
    pub timestamp: SystemTime,
}

/// The versions of the table that still have their files, oldest first.
pub async fn versions(dataset: &Dataset) -> Result<Vec<TableVersion>> {
    check_format(dataset)?;
    let mut versions: Vec<TableVersion> = dataset
        .versions()
        .await?
        .into_iter()
        .map(|version| TableVersion {
            version: version.version,
            timestamp: UNIX_EPOCH
                + Duration::from_millis(version.timestamp.timestamp_millis().max(0) as u64),
        })
        .collect();
    versions.sort_by_key(|version| version.version);
    Ok(versions)
}

/// The table at `version`, to read. A write to it fails, because it is not
/// the latest version.
pub async fn checkout(dataset: &Dataset, version: u64) -> Result<Dataset> {
    check_format(dataset)?;
    let old = dataset.checkout_version(version).await?;
    check_format(&old)?;
    Ok(old)
}

/// Make the content of `version` the latest version of the table, in a new
/// commit. The versions after it stay in the history.
#[tracing::instrument(skip_all, fields(uri = %dataset.uri(), version = version), err)]
pub async fn restore(dataset: &mut Dataset, version: u64) -> Result<()> {
    let mut old = checkout(dataset, version).await?;
    old.restore().await?;
    *dataset = old;
    Ok(())
}
