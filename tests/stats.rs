use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Float64Array, Int32Array, RecordBatch, StringArray,
    StructArray, TimestampNanosecondArray,
};
use arrow::datatypes::{DataType, Field, Float64Type, Int32Type, Schema, TimeUnit};
use lance_nd::encode::encode_rows;
use lance_nd::stats::{STATS_FIELD, add_stats, has_stats};
use nd_arrow_array::{Dimensions, NdArrowArray, NdRecordBatch};

/// One row with the given 1-D variables, each on its own dimension.
fn row(variables: Vec<(&str, ArrayRef)>) -> NdRecordBatch {
    let fields: Vec<Field> = variables
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
        .collect();
    let arrays = variables
        .into_iter()
        .map(|(name, array)| {
            let dims = Dimensions::try_new([(format!("{name}_dim"), array.len())]).unwrap();
            NdArrowArray::try_new(array, dims).unwrap()
        })
        .collect();
    NdRecordBatch::try_new_inferred(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn stats_of(rows: &[NdRecordBatch]) -> RecordBatch {
    add_stats(&encode_rows(rows).unwrap()).unwrap()
}

fn child<'a>(batch: &'a RecordBatch, variable: &str) -> &'a StructArray {
    batch
        .column_by_name(STATS_FIELD)
        .expect("a __stats__ column")
        .as_struct()
        .column_by_name(variable)
        .expect("a stats child")
        .as_struct()
}

fn min_max(child: &StructArray) -> (&ArrayRef, &ArrayRef) {
    (
        child.column_by_name("min").unwrap(),
        child.column_by_name("max").unwrap(),
    )
}

#[test]
fn min_max_per_row() {
    let batch = stats_of(&[
        row(vec![
            (
                "lat",
                Arc::new(Float64Array::from(vec![-90.0, -89.0, -88.0])),
            ),
            (
                "sst",
                Arc::new(Float64Array::from(vec![Some(5.0), None, Some(1.0)])),
            ),
        ]),
        row(vec![
            ("lat", Arc::new(Float64Array::from(vec![11.0, 10.0]))),
            ("sst", Arc::new(Float64Array::from(vec![None, Some(7.0)]))),
        ]),
    ]);
    let (min, max) = min_max(child(&batch, "lat"));
    assert_eq!(min.as_primitive::<Float64Type>().values(), &[-90.0, 10.0]);
    assert_eq!(max.as_primitive::<Float64Type>().values(), &[-88.0, 11.0]);
    let (min, max) = min_max(child(&batch, "sst"));
    assert_eq!(min.as_primitive::<Float64Type>().values(), &[1.0, 7.0]);
    assert_eq!(max.as_primitive::<Float64Type>().values(), &[5.0, 7.0]);
    assert!(has_stats(batch.schema().as_ref()));
}

#[test]
fn nan_and_all_null() {
    let batch = stats_of(&[
        row(vec![(
            "v",
            Arc::new(Float64Array::from(vec![f64::NAN, 2.0, f64::NAN])),
        )]),
        row(vec![(
            "v",
            Arc::new(Float64Array::from(vec![Some(f64::NAN), None])),
        )]),
    ]);
    let stats = child(&batch, "v");
    assert!(stats.is_valid(0) && stats.is_valid(1));
    let (min, max) = min_max(stats);
    assert_eq!(min.as_primitive::<Float64Type>().value(0), 2.0);
    assert_eq!(max.as_primitive::<Float64Type>().value(0), 2.0);
    assert!(min.is_null(1) && max.is_null(1));
}

#[test]
fn absent_variable_is_null_child() {
    let batch = stats_of(&[
        row(vec![(
            "pressure",
            Arc::new(Int32Array::from(vec![3, 1, 2])),
        )]),
        row(vec![("other", Arc::new(Int32Array::from(vec![0])))]),
    ]);
    let stats = child(&batch, "pressure");
    assert!(stats.is_valid(0) && stats.is_null(1));
    let (min, max) = min_max(stats);
    assert_eq!(min.as_primitive::<Int32Type>().value(0), 1);
    assert_eq!(max.as_primitive::<Int32Type>().value(0), 3);
}

#[test]
fn strings_and_timestamps() {
    let times = TimestampNanosecondArray::from(vec![30, 10, 20]).with_timezone("UTC");
    let batch = stats_of(&[row(vec![
        ("name", Arc::new(StringArray::from(vec!["b", "a", "c"]))),
        ("time", Arc::new(times)),
    ])]);
    let (min, max) = min_max(child(&batch, "name"));
    assert_eq!(min.as_string::<i32>().value(0), "a");
    assert_eq!(max.as_string::<i32>().value(0), "c");
    let (min, max) = min_max(child(&batch, "time"));
    assert_eq!(
        min.data_type(),
        &DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()))
    );
    assert_eq!(
        min.as_primitive::<arrow::datatypes::TimestampNanosecondType>()
            .value(0),
        10
    );
    assert_eq!(
        max.as_primitive::<arrow::datatypes::TimestampNanosecondType>()
            .value(0),
        30
    );
}

#[test]
fn other_types_get_no_child() {
    let batch = stats_of(&[row(vec![
        ("flag", Arc::new(BooleanArray::from(vec![true, false]))),
        ("v", Arc::new(Int32Array::from(vec![1]))),
    ])]);
    let stats = batch.column_by_name(STATS_FIELD).unwrap().as_struct();
    assert!(stats.column_by_name("flag").is_none());
    assert!(stats.column_by_name("v").is_some());

    let only_flags = stats_of(&[row(vec![(
        "flag",
        Arc::new(BooleanArray::from(vec![true])),
    )])]);
    assert!(only_flags.column_by_name(STATS_FIELD).is_none());
    assert!(has_stats(only_flags.schema().as_ref()));
}

mod tables {
    use super::*;
    use futures::{TryStreamExt, stream};
    use lance::Dataset;
    use lance_nd::{Error, NdWriter, WriteMode, WriteOptions, append, scan, variables, write};

    fn uri(dir: &tempfile::TempDir) -> String {
        dir.path().join("table.lance").to_str().unwrap().to_string()
    }

    fn lat_row(values: Vec<f64>) -> NdRecordBatch {
        row(vec![("lat", Arc::new(Float64Array::from(values)))])
    }

    fn rows(
        batches: Vec<NdRecordBatch>,
    ) -> impl futures::Stream<Item = lance_nd::Result<NdRecordBatch>> {
        stream::iter(batches.into_iter().map(Ok))
    }

    fn with_stats() -> WriteOptions {
        WriteOptions {
            stats: true,
            ..Default::default()
        }
    }

    async fn stats_column(dataset: &Dataset) -> RecordBatch {
        dataset
            .scan()
            .project(&[STATS_FIELD])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
    }

    fn table_has_stats(dataset: &Dataset) -> bool {
        has_stats(&Schema::from(dataset.schema()))
    }

    #[tokio::test]
    async fn writer_with_stats_stores_them() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = NdWriter::create(&uri(&dir), &with_stats()).await.unwrap();
        writer
            .write(rows(vec![lat_row(vec![3.0, 1.0]), lat_row(vec![-5.0])]))
            .await
            .unwrap();
        let dataset = writer.finish().await.unwrap();
        assert!(table_has_stats(&dataset));
        let batch = stats_column(&dataset).await;
        let (min, max) = min_max(child(&batch, "lat"));
        assert_eq!(min.as_primitive::<Float64Type>().values(), &[1.0, -5.0]);
        assert_eq!(max.as_primitive::<Float64Type>().values(), &[3.0, -5.0]);
    }

    #[tokio::test]
    async fn stats_off_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = NdWriter::create(&uri(&dir), &WriteOptions::default())
            .await
            .unwrap();
        writer.write(rows(vec![lat_row(vec![1.0])])).await.unwrap();
        let dataset = writer.finish().await.unwrap();
        assert!(!table_has_stats(&dataset));
        assert!(dataset.schema().field(STATS_FIELD).is_none());
    }

    #[tokio::test]
    async fn append_follows_the_table() {
        let dir = tempfile::tempdir().unwrap();
        let uri = uri(&dir);
        let mut writer = NdWriter::create(&uri, &with_stats()).await.unwrap();
        writer.write(rows(vec![lat_row(vec![1.0])])).await.unwrap();
        writer.finish().await.unwrap();
        let mut writer = NdWriter::append(&uri, &WriteOptions::default())
            .await
            .unwrap();
        writer
            .write(rows(vec![lat_row(vec![7.0, 8.0])]))
            .await
            .unwrap();
        let dataset = writer.finish().await.unwrap();
        let batch = stats_column(&dataset).await;
        let (min, _) = min_max(child(&batch, "lat"));
        assert_eq!(min.as_primitive::<Float64Type>().values(), &[1.0, 7.0]);

        let plain = dir.path().join("plain.lance").to_str().unwrap().to_string();
        let mut writer = NdWriter::create(&plain, &WriteOptions::default())
            .await
            .unwrap();
        writer.write(rows(vec![lat_row(vec![1.0])])).await.unwrap();
        writer.finish().await.unwrap();
        let mut writer = NdWriter::append(&plain, &with_stats()).await.unwrap();
        writer.write(rows(vec![lat_row(vec![2.0])])).await.unwrap();
        let dataset = writer.finish().await.unwrap();
        assert!(!table_has_stats(&dataset));
        assert!(dataset.schema().field(STATS_FIELD).is_none());
    }

    #[tokio::test]
    async fn write_fn_with_stats() {
        let dir = tempfile::tempdir().unwrap();
        let uri = uri(&dir);
        let options = WriteOptions {
            max_rows_per_commit: 1,
            ..with_stats()
        };
        let mut dataset = write(
            &uri,
            rows(vec![lat_row(vec![1.0]), lat_row(vec![2.0])]),
            &options,
        )
        .await
        .unwrap();
        append(
            &mut dataset,
            rows(vec![lat_row(vec![3.0])]),
            &WriteOptions::default(),
        )
        .await
        .unwrap();
        assert!(table_has_stats(&dataset));
        let batch = stats_column(&dataset).await;
        let (min, _) = min_max(child(&batch, "lat"));
        assert_eq!(min.as_primitive::<Float64Type>().values(), &[1.0, 2.0, 3.0]);

        let overwrite = WriteOptions {
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        let dataset = write(&uri, rows(vec![lat_row(vec![1.0])]), &overwrite)
            .await
            .unwrap();
        assert!(!table_has_stats(&dataset));
    }

    #[tokio::test]
    async fn new_variable_adds_a_child() {
        let dir = tempfile::tempdir().unwrap();
        let uri = uri(&dir);
        let mut writer = NdWriter::create(&uri, &with_stats()).await.unwrap();
        writer.write(rows(vec![lat_row(vec![1.0])])).await.unwrap();
        writer.finish().await.unwrap();
        let mut writer = NdWriter::append(&uri, &WriteOptions::default())
            .await
            .unwrap();
        let pressure = row(vec![("pressure", Arc::new(Int32Array::from(vec![4, 2])))]);
        writer.write(rows(vec![pressure])).await.unwrap();
        let dataset = writer.finish().await.unwrap();
        let batch = stats_column(&dataset).await;
        let stats = child(&batch, "pressure");
        assert!(stats.is_null(0) && stats.is_valid(1));
        assert_eq!(min_max(stats).0.as_primitive::<Int32Type>().value(1), 2);
        assert!(child(&batch, "lat").is_null(1));
    }

    #[tokio::test]
    async fn scan_hides_stats() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = NdWriter::create(&uri(&dir), &with_stats()).await.unwrap();
        writer
            .write(rows(vec![lat_row(vec![1.0, 2.0])]))
            .await
            .unwrap();
        let dataset = writer.finish().await.unwrap();
        let read: Vec<NdRecordBatch> = scan(&dataset)
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(read.len(), 1);
        let names: Vec<_> = read[0]
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["lat"]);
        let names: Vec<_> = variables(&dataset)
            .unwrap()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["lat"]);
        let error = scan(&dataset)
            .project(&[STATS_FIELD])
            .try_into_stream()
            .await
            .err();
        assert!(matches!(error, Some(Error::InvalidArgument(_))));
        let read: Vec<NdRecordBatch> = scan(&dataset)
            .project(&["lat"])
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(read[0].schema().fields().len(), 1);
    }

    #[test]
    fn stats_name_is_reserved() {
        let bad = row(vec![(STATS_FIELD, Arc::new(Int32Array::from(vec![1])))]);
        assert!(matches!(encode_rows(&[bad]), Err(Error::SchemaMismatch(_))));
    }
}
