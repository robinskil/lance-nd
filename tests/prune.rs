use arrow::datatypes::{DataType, Field, Fields, Schema};
use lance::deps::datafusion::common::Column;
use lance::deps::datafusion::logical_expr::{Expr, col, lit};
use lance_nd::Error;
use lance_nd::prune::prune_filter;
use lance_nd::stats::STATS_FIELD;

fn bounds(data_type: DataType) -> DataType {
    DataType::Struct(Fields::from(vec![
        Field::new("min", data_type.clone(), true),
        Field::new("max", data_type, true),
    ]))
}

/// Variables `lat`, `name`, `flag` and `a.b`. All but `flag` have stats.
fn table(with_stats: bool) -> Schema {
    let mut fields = vec![
        Field::new("lat", DataType::Float64, true),
        Field::new("name", DataType::Utf8, true),
        Field::new("flag", DataType::Boolean, true),
        Field::new("a.b", DataType::Int32, true),
        Field::new("__global_attrs__", DataType::Struct(Fields::empty()), true),
    ];
    if with_stats {
        let children = Fields::from(vec![
            Field::new("lat", bounds(DataType::Float64), true),
            Field::new("name", bounds(DataType::Utf8), true),
            Field::new("a.b", bounds(DataType::Int32), true),
        ]);
        fields.push(Field::new(STATS_FIELD, DataType::Struct(children), true));
    }
    Schema::new(fields)
}

fn prune(expr: Expr) -> lance_nd::Result<Option<Expr>> {
    prune_filter(&expr, &table(true))
}

#[test]
fn unknown_name_errors() {
    let error = prune(col("latt").gt(lit(1.0)));
    assert!(matches!(error, Err(Error::InvalidArgument(_))));
}

#[test]
fn reserved_names_error() {
    for name in [STATS_FIELD, "__global_attrs__"] {
        let expr = Expr::Column(Column::new_unqualified(name)).is_not_null();
        assert!(
            matches!(prune(expr), Err(Error::InvalidArgument(_))),
            "{name}"
        );
    }
}

#[test]
fn no_stats_column_keeps() {
    let result = prune_filter(&col("lat").gt(lit(1.0)), &table(false)).unwrap();
    assert!(result.is_none());
    let error = prune_filter(&col("latt").gt(lit(1.0)), &table(false));
    assert!(matches!(error, Err(Error::InvalidArgument(_))));
}

#[test]
fn supported_forms_rewrite() {
    let forms = [
        col("lat").eq(lit(1.0)),
        col("lat").lt(lit(1.0)),
        col("lat").lt_eq(lit(1.0)),
        col("lat").gt(lit(1.0)),
        col("lat").gt_eq(lit(1.0)),
        lit(1.0).lt(col("lat")),
        col("lat").between(lit(17.5), lit(25.4)),
        col("name").in_list(vec![lit("a"), lit("b")], false),
        col("lat").gt(lit(1.0)).and(col("name").eq(lit("x"))),
        col("lat").gt(lit(1.0)).or(col("name").eq(lit("x"))),
    ];
    for form in forms {
        let rewritten = prune(form.clone()).unwrap();
        let text = rewritten
            .unwrap_or_else(|| panic!("{form} keeps"))
            .to_string();
        assert!(text.contains(STATS_FIELD), "{form} -> {text}");
    }
}

#[test]
fn unsupported_forms_keep() {
    let forms = [
        !col("lat").gt(lit(1.0)),
        col("lat").not_eq(lit(1.0)),
        (col("lat") * lit(2.0)).gt(lit(1.0)),
        col("flag").eq(lit(true)),
        col("lat").not_between(lit(1.0), lit(2.0)),
        col("name").in_list(vec![lit("a")], true),
        col("name").like(lit("a%")),
        col("lat").gt(col("lat")),
    ];
    for form in forms {
        assert!(prune(form.clone()).unwrap().is_none(), "{form}");
    }
}

#[test]
fn and_drops_keep_side() {
    let both = col("lat")
        .gt(lit(1.0))
        .and((col("lat") * lit(2.0)).gt(lit(1.0)));
    assert_eq!(
        prune(both).unwrap(),
        prune(col("lat").gt(lit(1.0))).unwrap()
    );
}

#[test]
fn or_with_keep_side_keeps() {
    let either = col("lat")
        .gt(lit(1.0))
        .or((col("lat") * lit(2.0)).gt(lit(1.0)));
    assert!(prune(either).unwrap().is_none());
}

#[test]
fn dotted_name_matches() {
    let qualified = col("a.b").gt(lit(1));
    assert!(prune(qualified).unwrap().is_some());
    let plain = Expr::Column(Column::new_unqualified("a.b")).gt(lit(1));
    assert!(prune(plain).unwrap().is_some());
}
