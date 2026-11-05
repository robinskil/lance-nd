use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Float32Array, Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use futures::{StreamExt, stream};
use lance::Dataset;
use lance_nd::{
    ByteStreamSplit, ColumnOptions, Compression, Error, StructuralEncoding, WriteMode,
    WriteOptions, append, column_defaults, scan, set_column_defaults, variables, write,
};
use nd_arrow_array::{Attributes, Dimensions, NdArrowArray, NdRecordBatch};

fn dims(spec: &[(&str, usize)]) -> Dimensions {
    Dimensions::try_new(spec.iter().copied()).unwrap()
}

fn uri(dir: &tempfile::TempDir) -> String {
    dir.path().join("table.lance").to_str().unwrap().to_string()
}

fn rows(
    batches: Vec<NdRecordBatch>,
) -> impl futures::Stream<Item = lance_nd::Result<NdRecordBatch>> {
    stream::iter(batches.into_iter().map(Ok))
}

/// An xarray-like dataset: `time` and `lat` coordinates, a `(time, lat)`
/// data variable, a scalar and global attributes.
fn sea_surface(times: usize, lats: usize, base: f64, title: &str) -> NdRecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("time", DataType::Int32, true),
        Field::new("lat", DataType::Float64, true),
        Field::new("sst", DataType::Float64, true),
        Field::new("depth", DataType::Float32, true),
    ]));
    let time = NdArrowArray::try_new(
        Arc::new(Int32Array::from_iter_values(0..times as i32)),
        dims(&[("time", times)]),
    )
    .unwrap()
    .with_attributes(Attributes::new().with("units", "days since 2000-01-01"));
    let lat = NdArrowArray::try_new(
        Arc::new(Float64Array::from_iter_values(
            (0..lats).map(|i| -90.0 + i as f64),
        )),
        dims(&[("lat", lats)]),
    )
    .unwrap()
    .with_attributes(Attributes::new().with("units", "degrees_north"));
    // One missing element per dataset.
    let values = (0..times * lats).map(|i| (i != 1).then_some(base + i as f64));
    let sst = NdArrowArray::try_new(
        Arc::new(Float64Array::from_iter(values)),
        dims(&[("time", times), ("lat", lats)]),
    )
    .unwrap()
    .with_attributes(
        Attributes::new()
            .with("units", "K")
            .with("valid_range", vec![270.0f64, 310.0]),
    );
    let depth = NdArrowArray::try_new(
        Arc::new(Float32Array::from(vec![0.5])),
        Dimensions::scalar(),
    )
    .unwrap();
    NdRecordBatch::try_new_inferred(schema, vec![time, lat, sst, depth])
        .unwrap()
        .with_attributes(Attributes::new().with("title", title))
}

/// A dataset with other variables: one `(station)` variable.
fn stations(values: Vec<f32>) -> NdRecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "pressure",
        DataType::Float32,
        true,
    )]));
    let n = values.len();
    let pressure = NdArrowArray::try_new(
        Arc::new(Float32Array::from(values)),
        dims(&[("station", n)]),
    )
    .unwrap()
    .with_attributes(Attributes::new().with("units", "hPa"));
    NdRecordBatch::try_new_inferred(schema, vec![pressure]).unwrap()
}

/// Dataset `index` of a large collection: a subset of 20 variables, with
/// sizes and attribute keys that vary per dataset.
fn varied(index: usize) -> NdRecordBatch {
    let n = 1 + index % 5;
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    for v in 0..20 {
        if !(index + v).is_multiple_of(3) && v != index % 20 {
            continue;
        }
        let mut attrs = Attributes::new().with("units", format!("u{v}"));
        if index.is_multiple_of(4) {
            attrs.insert(format!("note_{}", index % 7), index as i64);
        }
        let array = NdArrowArray::try_new(
            Arc::new(Float64Array::from_iter_values(
                (0..n * 2).map(|i| (index * 100 + v * 10 + i) as f64),
            )),
            dims(&[("x", n), ("y", 2)]),
        )
        .unwrap()
        .with_attributes(attrs);
        fields.push(Field::new(format!("var{v:02}"), DataType::Float64, true));
        columns.push(array);
    }
    let mut batch =
        NdRecordBatch::try_new_inferred(Arc::new(Schema::new(fields)), columns).unwrap();
    if index.is_multiple_of(5) {
        batch = batch.with_attributes(Attributes::new().with("id", format!("ds{index}")));
    }
    batch
}

/// Compare by variable name: the table can store variables in another order.
fn assert_same(actual: &NdRecordBatch, expected: &NdRecordBatch) {
    let by_name = |b: &NdRecordBatch| -> BTreeMap<String, NdArrowArray> {
        b.schema()
            .fields()
            .iter()
            .zip(b.columns())
            .map(|(f, c)| (f.name().clone(), c.clone()))
            .collect()
    };
    let grid = |b: &NdRecordBatch| -> BTreeMap<String, usize> {
        b.target()
            .iter()
            .map(|d| (d.name().to_string(), d.size()))
            .collect()
    };
    assert_eq!(by_name(actual), by_name(expected));
    assert_eq!(grid(actual), grid(expected));
    assert_eq!(actual.attributes(), expected.attributes());
}

#[tokio::test]
async fn stream_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let a = sea_surface(3, 4, 280.0, "a");
    let b = stations(vec![1013.0, 1009.5, 998.25]);
    // A chunk is an nd record batch like any other row.
    let c = sea_surface(5, 7, 270.0, "c")
        .isel(&[("time", (0..2).into())])
        .unwrap();
    let dataset = write(
        &uri(&dir),
        rows(vec![a.clone(), b.clone(), c.clone()]),
        &WriteOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 3);
    let names: Vec<String> = variables(&dataset)
        .unwrap()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, vec!["time", "lat", "sst", "depth", "pressure"]);

    let dataset = Dataset::open(&uri(&dir)).await.unwrap();
    let mut stream = scan(&dataset).try_into_stream().await.unwrap();
    let mut read = Vec::new();
    while let Some(row) = stream.next().await {
        read.push(row.unwrap());
    }
    assert_eq!(read.len(), 3);
    assert_same(&read[0], &a);
    assert_same(&read[1], &b);
    assert_same(&read[2], &c);
}

#[tokio::test]
async fn many_datasets_with_varying_variables() {
    let dir = tempfile::tempdir().unwrap();
    let expected: Vec<NdRecordBatch> = (0..500).map(varied).collect();
    let options = WriteOptions {
        max_rows_per_group: 64,
        ..Default::default()
    };
    let dataset = write(&uri(&dir), rows(expected.clone()), &options)
        .await
        .unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 500);
    assert_eq!(variables(&dataset).unwrap().len(), 20);

    let read = scan(&dataset).try_collect().await.unwrap();
    assert_eq!(read.len(), 500);
    for (actual, expected) in read.iter().zip(&expected) {
        assert_same(actual, expected);
    }

    // A projection reads only the named variables that each row has.
    let projected = scan(&dataset)
        .project(&["var03", "var17"])
        .try_collect()
        .await
        .unwrap();
    assert_eq!(projected.len(), 500);
    for (actual, expected) in projected.iter().zip(&expected) {
        let names: Vec<&str> = actual
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        let wanted: Vec<&str> = ["var03", "var17"]
            .into_iter()
            .filter(|n| expected.column_by_name(n).is_some())
            .collect();
        assert_eq!(names, wanted);
        for name in names {
            assert_eq!(actual.column_by_name(name), expected.column_by_name(name));
        }
        assert_eq!(actual.attributes(), expected.attributes());
    }
}

#[tokio::test]
async fn append_extends_the_schema() {
    let dir = tempfile::tempdir().unwrap();
    let first = stations(vec![1013.0, 1009.5]);
    let mut dataset = write(
        &uri(&dir),
        rows(vec![first.clone()]),
        &WriteOptions::default(),
    )
    .await
    .unwrap();
    let version = dataset.version().version;

    // A new attribute key, new variables and new global attributes.
    let schema = Arc::new(Schema::new(vec![Field::new(
        "pressure",
        DataType::Float32,
        true,
    )]));
    let pressure = NdArrowArray::try_new(
        Arc::new(Float32Array::from(vec![1000.0, 1001.0])),
        dims(&[("station", 2)]),
    )
    .unwrap()
    .with_attributes(
        Attributes::new()
            .with("units", "hPa")
            .with("long_name", "air pressure"),
    );
    let second = NdRecordBatch::try_new_inferred(schema, vec![pressure]).unwrap();
    let third = sea_surface(2, 2, 300.0, "third");
    append(
        &mut dataset,
        rows(vec![second.clone(), third.clone()]),
        &WriteOptions::default(),
    )
    .await
    .unwrap();
    assert!(dataset.version().version > version);

    // Append mode on an existing uri appends too.
    let fourth = stations(vec![990.0]);
    let options = WriteOptions {
        mode: WriteMode::Append,
        ..Default::default()
    };
    write(&uri(&dir), rows(vec![fourth.clone()]), &options)
        .await
        .unwrap();

    let dataset = Dataset::open(&uri(&dir)).await.unwrap();
    let read = scan(&dataset).try_collect().await.unwrap();
    assert_eq!(read.len(), 4);
    assert_same(&read[0], &first);
    assert_same(&read[1], &second);
    assert_same(&read[2], &third);
    assert_same(&read[3], &fourth);
}

#[tokio::test]
async fn append_rejects_a_type_change() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = write(
        &uri(&dir),
        rows(vec![stations(vec![1.0])]),
        &WriteOptions::default(),
    )
    .await
    .unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "pressure",
        DataType::Float64,
        true,
    )]));
    let pressure = NdArrowArray::try_new(
        Arc::new(Float64Array::from(vec![1000.0])),
        dims(&[("station", 1)]),
    )
    .unwrap();
    let batch = NdRecordBatch::try_new_inferred(schema, vec![pressure]).unwrap();
    let err = append(&mut dataset, rows(vec![batch]), &WriteOptions::default())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("pressure"), "{err}");
}

#[tokio::test]
async fn scan_filters_and_limits() {
    let dir = tempfile::tempdir().unwrap();
    let batches = vec![
        sea_surface(2, 3, 280.0, "a"),
        sea_surface(2, 3, 290.0, "b"),
        stations(vec![1.0, 2.0]),
    ];
    let dataset = write(&uri(&dir), rows(batches.clone()), &WriteOptions::default())
        .await
        .unwrap();

    let filtered = scan(&dataset)
        .project(&["sst"])
        .filter("__global_attrs__.title = 'b'")
        .try_collect()
        .await
        .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].num_columns(), 1);
    assert_eq!(
        filtered[0].column(0),
        batches[1].column_by_name("sst").unwrap()
    );

    let limited = scan(&dataset).limit(2).try_collect().await.unwrap();
    assert_eq!(limited.len(), 2);

    assert!(
        scan(&dataset)
            .project(&["nope"])
            .try_collect()
            .await
            .is_err()
    );
}

#[tokio::test]
async fn reserved_names_and_empty_streams_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "__global_attrs__",
        DataType::Int32,
        true,
    )]));
    let column =
        NdArrowArray::try_new(Arc::new(Int32Array::from(vec![1])), dims(&[("x", 1)])).unwrap();
    let batch = NdRecordBatch::try_new_inferred(schema, vec![column]).unwrap();
    assert!(
        write(&uri(&dir), rows(vec![batch]), &WriteOptions::default())
            .await
            .is_err()
    );
    assert!(
        write(&uri(&dir), rows(vec![]), &WriteOptions::default())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn stream_errors_stop_the_write() {
    let dir = tempfile::tempdir().unwrap();
    let input = stream::iter(vec![
        Ok(stations(vec![1.0])),
        Err(lance_nd::Error::InvalidArgument("source failed".into())),
    ]);
    let err = write(&uri(&dir), input, &WriteOptions::default())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("source failed"));
}

const COMPRESSION: &str = "lance-encoding:compression";

/// The values of metadata `key` on the leaf fields of `field`.
fn leaf_metadata(field: &Field, key: &str) -> Vec<Option<String>> {
    match field.data_type() {
        DataType::Struct(children) => children
            .iter()
            .flat_map(|child| leaf_metadata(child, key))
            .collect(),
        DataType::List(item) => leaf_metadata(item, key),
        _ => vec![field.metadata().get(key).cloned()],
    }
}

/// Assert that each leaf of column `name` has `value` for metadata `key`.
fn assert_leaves(dataset: &Dataset, name: &str, key: &str, value: &str) {
    let schema = Schema::from(dataset.schema());
    let leaves = leaf_metadata(schema.field_with_name(name).unwrap(), key);
    assert!(!leaves.is_empty());
    assert!(
        leaves.iter().all(|v| v.as_deref() == Some(value)),
        "{name}: {key} is {leaves:?}"
    );
}

fn root_metadata(dataset: &Dataset, name: &str, key: &str) -> Option<String> {
    let schema = Schema::from(dataset.schema());
    schema
        .field_with_name(name)
        .unwrap()
        .metadata()
        .get(key)
        .cloned()
}

/// The size of the data files of the table.
fn data_size(dir: &tempfile::TempDir) -> u64 {
    std::fs::read_dir(dir.path().join("table.lance").join("data"))
        .unwrap()
        .map(|entry| entry.unwrap().metadata().unwrap().len())
        .sum()
}

#[tokio::test]
async fn column_options_are_stored_in_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let defaults = ColumnOptions::default()
        .with_compression(Compression::Zstd { level: Some(3) })
        .with_byte_stream_split(ByteStreamSplit::On);
    let sst = ColumnOptions::default()
        .with_compression(Compression::None)
        .with_structural_encoding(StructuralEncoding::FullZip);
    let options = WriteOptions {
        column_defaults: Some(defaults.clone()),
        column_options: [("sst".to_string(), sst)].into(),
        ..WriteOptions::default()
    };
    let a = sea_surface(3, 4, 280.0, "a");
    let mut dataset = write(&uri(&dir), rows(vec![a.clone()]), &options)
        .await
        .unwrap();
    assert_eq!(column_defaults(&dataset).unwrap(), defaults);

    assert_leaves(&dataset, "lat", COMPRESSION, "zstd");
    assert_leaves(&dataset, "lat", "lance-encoding:compression-level", "3");
    assert_leaves(&dataset, "lat", "lance-encoding:bss", "on");
    assert_eq!(
        root_metadata(&dataset, "lat", "lance-encoding:structural-encoding"),
        None
    );
    assert_leaves(&dataset, "sst", COMPRESSION, "none");
    assert_leaves(&dataset, "sst", "lance-encoding:bss", "on");
    assert_eq!(
        root_metadata(&dataset, "sst", "lance-encoding:structural-encoding").as_deref(),
        Some("fullzip")
    );

    // A write without options gives the stored defaults to a new column.
    let b = stations(vec![1013.0, 1009.5]);
    append(
        &mut dataset,
        rows(vec![b.clone()]),
        &WriteOptions::default(),
    )
    .await
    .unwrap();
    assert_leaves(&dataset, "pressure", COMPRESSION, "zstd");

    let read = scan(&dataset).try_collect().await.unwrap();
    assert_same(&read[0], &a);
    assert_same(&read[1], &b);
    let lance_key = |key: &String| key.starts_with("lance-encoding:");
    for field in variables(&dataset).unwrap() {
        assert!(!field.metadata().keys().any(lance_key), "{field:?}");
    }
    for field in read.iter().flat_map(|row| row.schema().fields().to_vec()) {
        assert!(!field.metadata().keys().any(lance_key), "{field:?}");
    }
}

#[tokio::test]
async fn column_defaults_change_only_on_request() {
    let dir = tempfile::tempdir().unwrap();
    let zstd = ColumnOptions::default().with_compression(Compression::Zstd { level: None });
    let lz4 = ColumnOptions::default().with_compression(Compression::Lz4);
    let options = WriteOptions {
        column_defaults: Some(zstd.clone()),
        ..WriteOptions::default()
    };
    let mut dataset = write(
        &uri(&dir),
        rows(vec![sea_surface(2, 2, 280.0, "a")]),
        &options,
    )
    .await
    .unwrap();

    // Equal defaults are accepted, other defaults are rejected.
    append(
        &mut dataset,
        rows(vec![sea_surface(2, 2, 281.0, "b")]),
        &options,
    )
    .await
    .unwrap();
    let other = WriteOptions {
        column_defaults: Some(lz4.clone()),
        ..WriteOptions::default()
    };
    let error = append(&mut dataset, rows(vec![stations(vec![1.0])]), &other)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidArgument(_)), "{error}");

    set_column_defaults(&mut dataset, &lz4).await.unwrap();
    append(&mut dataset, rows(vec![stations(vec![1.0])]), &other)
        .await
        .unwrap();
    let dataset = Dataset::open(&uri(&dir)).await.unwrap();
    assert_eq!(column_defaults(&dataset).unwrap(), lz4);
    assert_leaves(&dataset, "pressure", COMPRESSION, "lz4");
    assert_leaves(&dataset, "sst", COMPRESSION, "zstd");
    assert_eq!(dataset.count_rows(None).await.unwrap(), 3);

    let mut dataset = dataset;
    set_column_defaults(&mut dataset, &ColumnOptions::default())
        .await
        .unwrap();
    assert_eq!(column_defaults(&dataset).unwrap(), ColumnOptions::default());
    assert!(
        !dataset
            .schema()
            .metadata
            .keys()
            .any(|key| key.starts_with("lance_nd.column_defaults."))
    );
}

#[tokio::test]
async fn compression_changes_the_stored_size() {
    let field = |row: usize| {
        let n = 50_000;
        let values = (0..n).map(|i| ((i * 7919 + row) % 97) as f64 * 1.37);
        let array = NdArrowArray::try_new(
            Arc::new(Float64Array::from_iter_values(values)),
            dims(&[("x", n)]),
        )
        .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "field",
            DataType::Float64,
            true,
        )]));
        NdRecordBatch::try_new_inferred(schema, vec![array]).unwrap()
    };
    let mut sizes = Vec::new();
    for compression in [Compression::None, Compression::Zstd { level: None }] {
        let dir = tempfile::tempdir().unwrap();
        let options = WriteOptions {
            column_defaults: Some(ColumnOptions::default().with_compression(compression)),
            ..WriteOptions::default()
        };
        write(&uri(&dir), rows((0..4).map(field).collect()), &options)
            .await
            .unwrap();
        sizes.push(data_size(&dir));
    }
    assert!(sizes[1] * 4 < sizes[0], "sizes: {sizes:?}");
}

#[tokio::test]
async fn empty_projection_reads_global_attributes() {
    let dir = tempfile::tempdir().unwrap();
    let batches = vec![sea_surface(2, 3, 280.0, "a"), sea_surface(2, 3, 290.0, "b")];
    let dataset = write(&uri(&dir), rows(batches), &WriteOptions::default())
        .await
        .unwrap();
    let rows = scan(&dataset).project(&[]).try_collect().await.unwrap();
    assert_eq!(rows.len(), 2);
    for (row, title) in rows.iter().zip(["a", "b"]) {
        assert_eq!(row.num_columns(), 0);
        let expected = Attributes::new().with("title", title);
        assert_eq!(row.attributes(), &expected);
    }
}

#[tokio::test]
async fn empty_projection_without_global_attributes() {
    let dir = tempfile::tempdir().unwrap();
    let batches = vec![
        stations(vec![1.0]),
        stations(vec![2.0, 3.0]),
        stations(vec![4.0]),
    ];
    let dataset = write(&uri(&dir), rows(batches), &WriteOptions::default())
        .await
        .unwrap();
    let rows = scan(&dataset).project(&[]).try_collect().await.unwrap();
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|row| row.num_columns() == 0));
    let filtered = scan(&dataset)
        .project(&[])
        .filter("array_length(pressure.values) > 1")
        .try_collect()
        .await
        .unwrap();
    assert_eq!(filtered.len(), 1);
}

#[tokio::test]
async fn projection_takes_whole_variables() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = write(
        &uri(&dir),
        rows(vec![sea_surface(2, 3, 280.0, "a")]),
        &WriteOptions::default(),
    )
    .await
    .unwrap();
    for name in [
        "sst.attrs.units",
        "sst.values",
        "sst.attrs",
        "__global_attrs__.title",
        "__global_attrs__",
        "nope",
    ] {
        let error = scan(&dataset).project(&[name]).try_collect().await.err();
        match error {
            Some(Error::InvalidArgument(message)) => {
                assert!(message.contains(name), "{name}: {message}")
            }
            other => panic!("{name}: {other:?}"),
        }
    }
    let error = scan(&dataset)
        .project(&["sst.attrs.units"])
        .try_collect()
        .await
        .err();
    let message = error.unwrap().to_string();
    assert!(message.contains("whole"), "{message}");
}
