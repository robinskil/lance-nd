//! Storage options for tables on object stores, such as S3.
//!
//! The options are the Lance storage options, for example `aws_region`,
//! `aws_endpoint` or `allow_http`. An empty map uses the defaults of the
//! environment. Each object store needs its cargo feature: `aws`, `gcp` or
//! `azure`.

use std::collections::HashMap;
use std::sync::Arc;

use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};

/// Lance storage options, by key, for tables on object stores.
///
/// The keys are the Lance storage options, for example `aws_region`,
/// `aws_endpoint` or `allow_http`. An empty map uses the defaults of the
/// environment. Each object store needs its cargo feature: `aws`, `gcp` or
/// `azure`.
pub type StorageOptions = HashMap<String, String>;

/// The object store parameters for `options`, or `None` for the defaults.
pub(crate) fn store_params(options: &StorageOptions) -> Option<ObjectStoreParams> {
    if options.is_empty() {
        return None;
    }
    Some(ObjectStoreParams {
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            options.clone(),
        ))),
        ..Default::default()
    })
}

/// Open the Lance dataset at `uri` with `options`.
pub(crate) async fn open_dataset(uri: &str, options: &StorageOptions) -> lance::Result<Dataset> {
    DatasetBuilder::from_uri(uri)
        .with_storage_options(options.clone())
        .load()
        .await
}
