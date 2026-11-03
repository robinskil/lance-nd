use arrow::datatypes::{DataType, Field, Fields, Schema};
use lance::deps::datafusion::common::Column;
use lance::deps::datafusion::functions::core::expr_fn::{get_field, get_field_path};
use lance::deps::datafusion::logical_expr::{Expr, col, lit};
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

/// The values of `variable`: `<variable>.values`.
fn values(variable: &str) -> Expr {
    get_field(col(Column::new_unqualified(variable)), "values")
}

#[test]
fn names_are_not_checked() {
    assert!(prune(values("latt").gt(lit(1.0))).unwrap().is_none());
    for form in [
        units("latt").eq(lit("K")),
        col(STATS_FIELD).is_not_null(),
        col("lat").gt(lit(1.0)),
    ] {
        assert_eq!(prune(form.clone()).unwrap(), Some(form.clone()), "{form}");
    }
}

#[test]
fn no_stats_column_keeps() {
    let result = prune_filter(&values("lat").gt(lit(1.0)), &table(false)).unwrap();
    assert!(result.is_none());
}

#[test]
fn supported_forms_rewrite() {
    let forms = [
        values("lat").eq(lit(1.0)),
        values("lat").lt(lit(1.0)),
        values("lat").lt_eq(lit(1.0)),
        values("lat").gt(lit(1.0)),
        values("lat").gt_eq(lit(1.0)),
        lit(1.0).lt(values("lat")),
        values("lat").between(lit(17.5), lit(25.4)),
        values("name").in_list(vec![lit("a"), lit("b")], false),
        values("lat").gt(lit(1.0)).and(values("name").eq(lit("x"))),
        values("lat").gt(lit(1.0)).or(values("name").eq(lit("x"))),
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
        !values("lat").gt(lit(1.0)),
        values("lat").not_eq(lit(1.0)),
        (values("lat") * lit(2.0)).gt(lit(1.0)),
        values("flag").eq(lit(true)),
        values("lat").not_between(lit(1.0), lit(2.0)),
        values("name").in_list(vec![lit("a")], true),
        values("name").like(lit("a%")),
        values("lat").gt(values("lat")),
    ];
    for form in forms {
        assert!(prune(form.clone()).unwrap().is_none(), "{form}");
    }
}

#[test]
fn and_drops_keep_side() {
    let both = values("lat")
        .gt(lit(1.0))
        .and((values("lat") * lit(2.0)).gt(lit(1.0)));
    assert_eq!(
        prune(both).unwrap(),
        prune(values("lat").gt(lit(1.0))).unwrap()
    );
}

#[test]
fn or_with_keep_side_keeps() {
    let either = values("lat")
        .gt(lit(1.0))
        .or((values("lat") * lit(2.0)).gt(lit(1.0)));
    assert!(prune(either).unwrap().is_none());
}

#[test]
fn dotted_name_matches() {
    let qualified = get_field(col("a.b"), "values").gt(lit(1));
    assert!(prune(qualified).unwrap().is_some());
    let plain = values("a.b").gt(lit(1));
    assert!(prune(plain).unwrap().is_some());
}

#[test]
fn uncomparable_literal_keeps() {
    for form in [
        values("lat").gt(lit("abc")),
        values("lat").eq(lit(true)),
        values("name").lt(lit(1.0)),
        values("lat").between(lit(1.0), lit("z")),
        values("lat").in_list(vec![lit(1.0), lit("z")], false),
    ] {
        assert!(prune(form.clone()).unwrap().is_none(), "{form}");
    }
    assert!(prune(values("lat").gt(lit(1i64))).unwrap().is_some());
}

fn global(key: &str) -> Expr {
    get_field(col("__global_attrs__"), key)
}

fn units(variable: &str) -> Expr {
    get_field(get_field(col(variable), "attrs"), "units")
}

#[test]
fn parts_without_values_stay() {
    let forms = [
        global("id").eq(lit("a")),
        units("lat").eq(lit("K")),
        get_field_path(col("lat"), vec![lit("attrs"), lit("units")]).eq(lit("K")),
        units("flag").is_null(),
        !global("id").eq(lit("a")),
        global("id").like(lit("a%")),
        col("__global_attrs__").is_not_null(),
        get_field(col("lat"), "dim_sizes").is_not_null(),
        col("name").is_null(),
    ];
    for form in forms {
        assert_eq!(prune(form.clone()).unwrap(), Some(form.clone()), "{form}");
        assert_eq!(
            prune_filter(&form, &table(false)).unwrap(),
            Some(form.clone()),
            "{form} without stats"
        );
    }
}

#[test]
fn mixed_or_keeps_both_sides() {
    let attribute = global("id").eq(lit("a"));
    let mixed = attribute.clone().or(values("lat").gt(lit(1.0)));
    let stats = prune(values("lat").gt(lit(1.0))).unwrap().unwrap();
    assert_eq!(prune(mixed).unwrap(), Some(attribute.or(stats)));
}

#[test]
fn mixed_and_drops_keep_side() {
    let attribute = global("id").eq(lit("a"));
    let mixed = attribute.clone().and(values("lat").not_eq(lit(1.0)));
    assert_eq!(prune(mixed).unwrap(), Some(attribute));
}

#[test]
fn mixed_or_with_keep_side_keeps() {
    let mixed = global("id").eq(lit("a")).or(values("lat").not_eq(lit(1.0)));
    assert!(prune(mixed).unwrap().is_none());
}

#[test]
fn not_over_values_keeps() {
    for form in [
        !values("lat").gt(lit(1.0)),
        !(global("id").eq(lit("a")).and(values("lat").gt(lit(1.0)))),
    ] {
        assert!(prune(form.clone()).unwrap().is_none(), "{form}");
    }
}

#[test]
fn values_without_stats_form_keep() {
    for form in [
        values("lat").is_not_null(),
        get_field_path(col("lat"), vec![lit("values")]).is_null(),
    ] {
        assert!(prune(form.clone()).unwrap().is_none(), "{form}");
    }
}
