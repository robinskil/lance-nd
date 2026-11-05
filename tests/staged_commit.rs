//! Staged writes with one commit, for `NdWriter`.
//!
//! Finding for Lance 11:
//! - Keep one Lance schema for all staged writes. For each batch, merge the
//!   batch schema into it and call `set_field_id` with the manifest's
//!   `max_field_id`, so new fields get new ids and old ids stay.
//! - Write each batch with `FragmentCreateBuilder` and the merged schema
//!   projected to the batch columns. The data files then use the merged ids,
//!   also for new children of an existing struct.
//! - Give the staged fragments ids after the manifest's `max_fragment_id`.
//! - New table: commit one `Operation::Overwrite` with the merged schema.
//! - Existing table: commit one `Operation::Merge` with the old and the
//!   staged fragments and `preserves_nullability: true`. Old fragments read
//!   the new fields as null.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float64Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
    StructArray,
};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use lance::Dataset;
use lance::dataset::fragment::write::FragmentCreateBuilder;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{CommitBuilder, WriteMode, WriteParams};
use lance::table::format::Fragment;
use lance_core::datatypes::{OnMissing, OnTypeMismatch, Schema as LanceSchema};
use lance_file::version::LanceFileVersion;

fn uri(dir: &tempfile::TempDir) -> String {
    dir.path().join("table.lance").to_str().unwrap().to_string()
}

fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
        .collect();
    let arrays = columns.into_iter().map(|(_, array)| array).collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn int(values: &[i32]) -> ArrayRef {
    Arc::new(Int32Array::from(values.to_vec()))
}

fn float(values: &[f64]) -> ArrayRef {
    Arc::new(Float64Array::from(values.to_vec()))
}

/// Data files written for one future commit.
struct Staged {
    uri: String,
    base: Option<Dataset>,
    schema: Option<lance_core::datatypes::Schema>,
    fragments: Vec<Fragment>,
}

impl Staged {
    async fn new(uri: &str) -> Self {
        let base = Dataset::open(uri).await.ok();
        let schema = base.as_ref().map(|d| d.schema().clone());
        Self {
            uri: uri.to_string(),
            base,
            schema,
            fragments: Vec::new(),
        }
    }

    /// Grow the schema, then write the batch with the field ids of that schema.
    async fn write(&mut self, batch: RecordBatch) {
        let arrow = batch.schema();
        let mut merged = match &self.schema {
            Some(schema) => schema.merge(arrow.as_ref()).unwrap(),
            None => LanceSchema::try_from(arrow.as_ref()).unwrap(),
        };
        let max_existing = self.base.as_ref().map(|d| d.manifest.max_field_id());
        merged.set_field_id(max_existing);
        let projected = merged
            .project_by_schema(arrow.as_ref(), OnMissing::Error, OnTypeMismatch::Error)
            .unwrap();
        let params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };
        let reader = RecordBatchIterator::new([Ok(batch)], arrow.clone());
        let fragments = FragmentCreateBuilder::new(&self.uri)
            .schema(&projected)
            .write_params(&params)
            .write_fragments(reader)
            .await
            .unwrap();
        self.fragments.extend(fragments);
        self.schema = Some(merged);
    }

    async fn commit(self) -> Dataset {
        let schema = self.schema.unwrap();
        let first_id = self
            .base
            .as_ref()
            .and_then(|d| d.manifest.max_fragment_id())
            .map_or(0, |id| id + 1);
        let mut fragments = self.fragments;
        for (id, fragment) in (first_id..).zip(&mut fragments) {
            fragment.id = id;
        }
        match self.base {
            None => {
                let operation = Operation::Overwrite {
                    fragments,
                    schema,
                    config_upsert_values: None,
                    initial_bases: None,
                };
                CommitBuilder::new(self.uri.as_str())
                    .with_storage_format(LanceFileVersion::V2_2)
                    .execute(Transaction::new(0, operation, None))
                    .await
                    .unwrap()
            }
            Some(base) => {
                let mut all: Vec<Fragment> = base
                    .get_fragments()
                    .iter()
                    .map(|f| f.metadata().clone())
                    .collect();
                all.extend(fragments);
                let operation = Operation::Merge {
                    fragments: all,
                    schema,
                    preserves_nullability: true,
                };
                let version = base.version().version;
                CommitBuilder::new(Arc::new(base))
                    .execute(Transaction::new(version, operation, None))
                    .await
                    .unwrap()
            }
        }
    }
}

#[tokio::test]
async fn staged_create_with_new_column() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let mut staged = Staged::new(&uri).await;
    staged.write(batch(vec![("a", int(&[1]))])).await;
    // Other column order: field ids must follow the names, not the positions.
    staged
        .write(batch(vec![("b", float(&[2.5])), ("a", int(&[2]))]))
        .await;
    assert!(Dataset::open(&uri).await.is_err());

    let dataset = staged.commit().await;
    assert_eq!(dataset.version().version, 1);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 2);
    let rows = dataset.scan().try_into_batch().await.unwrap();
    let a = rows.column_by_name("a").unwrap();
    let b = rows.column_by_name("b").unwrap();
    assert_eq!(a.as_ref(), int(&[1, 2]).as_ref());
    assert!(b.is_null(0));
    assert_eq!(
        b.as_any().downcast_ref::<Float64Array>().unwrap().value(1),
        2.5
    );
}

#[tokio::test]
async fn staged_append_with_new_column() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let first = batch(vec![("a", int(&[1]))]);
    let schema = first.schema();
    let params = WriteParams {
        data_storage_version: Some(LanceFileVersion::V2_2),
        ..Default::default()
    };
    Dataset::write(
        RecordBatchIterator::new([Ok(first)], schema),
        &uri,
        Some(params),
    )
    .await
    .unwrap();

    let mut staged = Staged::new(&uri).await;
    staged
        .write(batch(vec![("a", int(&[2])), ("b", float(&[2.5]))]))
        .await;
    let before = Dataset::open(&uri).await.unwrap();
    assert_eq!(before.version().version, 1);
    assert_eq!(before.count_rows(None).await.unwrap(), 1);

    let dataset = staged.commit().await;
    assert_eq!(dataset.version().version, 2);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 2);
    let rows = dataset.scan().try_into_batch().await.unwrap();
    let b = rows.column_by_name("b").unwrap();
    assert_eq!(
        rows.column_by_name("a").unwrap().as_ref(),
        int(&[1, 2]).as_ref()
    );
    assert!(b.is_null(0));
    assert!(b.is_valid(1));
}

#[tokio::test]
async fn staged_nested_struct_child() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let x = Field::new("x", DataType::Int32, true);
    let y = Field::new("y", DataType::Utf8, true);
    let narrow = StructArray::new(Fields::from(vec![x.clone()]), vec![int(&[1])], None);
    let wide = StructArray::new(
        Fields::from(vec![x, y]),
        vec![int(&[2]), Arc::new(StringArray::from(vec!["k"]))],
        None,
    );
    let mut staged = Staged::new(&uri).await;
    staged
        .write(batch(vec![("s", Arc::new(narrow)), ("a", int(&[1]))]))
        .await;
    staged
        .write(batch(vec![("s", Arc::new(wide)), ("a", int(&[2]))]))
        .await;

    let dataset = staged.commit().await;
    let rows = dataset.scan().try_into_batch().await.unwrap();
    let s = rows.column_by_name("s").unwrap();
    let s = s.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(
        s.column_by_name("x").unwrap().as_ref(),
        int(&[1, 2]).as_ref()
    );
    let y = s.column_by_name("y").unwrap();
    assert!(y.is_null(0));
    assert_eq!(
        y.as_any().downcast_ref::<StringArray>().unwrap().value(1),
        "k"
    );
    assert_eq!(
        rows.column_by_name("a").unwrap().as_ref(),
        int(&[1, 2]).as_ref()
    );
}
