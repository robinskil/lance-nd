//! The format version of a lance-nd table.
//!
//! A table has the schema metadata key [`FORMAT_METADATA_KEY`] with the
//! value [`FORMAT_VERSION`]. lance-nd reads and changes only tables with this
//! key and value. A change of the table layout gets a new version.

use std::collections::HashMap;

use lance::Dataset;

use crate::error::{Error, Result};
use crate::store::{StorageOptions, open_dataset};

/// The table schema metadata key of the format version.
pub const FORMAT_METADATA_KEY: &str = "lance_nd:format";

/// The format version that this release reads and writes.
pub const FORMAT_VERSION: &str = "1";

/// Open the lance-nd table at `uri`. Fails when no table is there, or when
/// the table has another format version.
pub async fn open(uri: &str) -> Result<Dataset> {
    open_with(uri, &StorageOptions::new()).await
}

/// Open the lance-nd table at `uri` with Lance storage options, for example
/// the credentials of an object store. See [`crate::store`].
#[tracing::instrument(skip_all, fields(uri = %uri), err)]
pub async fn open_with(uri: &str, options: &StorageOptions) -> Result<Dataset> {
    let dataset = open_dataset(uri, options).await?;
    check_format(&dataset)?;
    Ok(dataset)
}

/// Fails when `dataset` is not a lance-nd table of [`FORMAT_VERSION`].
pub fn check_format(dataset: &Dataset) -> Result<()> {
    match dataset.schema().metadata.get(FORMAT_METADATA_KEY) {
        Some(version) if version == FORMAT_VERSION => Ok(()),
        Some(version) => Err(Error::InvalidTable(format!(
            "the table has lance-nd format {version}, this release reads format {FORMAT_VERSION}"
        ))),
        None => Err(Error::InvalidTable(format!(
            "the table has no {FORMAT_METADATA_KEY} key: it is not a lance-nd table of format {FORMAT_VERSION}"
        ))),
    }
}

/// Add the format key to the metadata of a new table.
pub(crate) fn add_format(metadata: &mut HashMap<String, String>) {
    metadata.insert(FORMAT_METADATA_KEY.to_string(), FORMAT_VERSION.to_string());
}
