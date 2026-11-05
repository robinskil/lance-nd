//! Property: a stats filter never skips a row that matches the filter.
//!
//! A row matches a value part when one of its values satisfies it. Null and
//! NaN values never match, as the stats rules say.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array};
use arrow::datatypes::{Field, Int64Type, Schema};
use futures::stream;
use lance_nd::datafusion::functions::core::expr_fn::get_field;
use lance_nd::datafusion::logical_expr::{Expr, col, lit};
use lance_nd::{WriteOptions, scan, write};
use nd_arrow_array::{Attributes, Dimensions, NdArrowArray, NdRecordBatch};
use proptest::prelude::*;

/// A value part of a filter on `v.values`.
#[derive(Debug, Clone)]
enum Part {
    Eq(f64),
    Lt(f64),
    LtEq(f64),
    Gt(f64),
    GtEq(f64),
    Between(f64, f64),
    In(Vec<f64>),
}

impl Part {
    fn holds(&self, v: f64) -> bool {
        match self {
            Part::Eq(x) => v == *x,
            Part::Lt(x) => v < *x,
            Part::LtEq(x) => v <= *x,
            Part::Gt(x) => v > *x,
            Part::GtEq(x) => v >= *x,
            Part::Between(a, b) => *a <= v && v <= *b,
            Part::In(list) => list.contains(&v),
        }
    }

    fn expr(&self) -> Expr {
        let values = get_field(col("v"), "values");
        match self {
            Part::Eq(x) => values.eq(lit(*x)),
            Part::Lt(x) => values.lt(lit(*x)),
            Part::LtEq(x) => values.lt_eq(lit(*x)),
            Part::Gt(x) => values.gt(lit(*x)),
            Part::GtEq(x) => values.gt_eq(lit(*x)),
            Part::Between(a, b) => values.between(lit(*a), lit(*b)),
            Part::In(list) => values.in_list(list.iter().map(|x| lit(*x)).collect(), false),
        }
    }
}

/// A filter: one part, or two parts with AND or OR.
#[derive(Debug, Clone)]
enum Filter {
    One(Part),
    And(Part, Part),
    Or(Part, Part),
}

impl Filter {
    /// `true` when a row with `values` can match.
    fn matches(&self, values: &[Option<f64>]) -> bool {
        let any = |part: &Part| {
            values
                .iter()
                .flatten()
                .any(|v| !v.is_nan() && part.holds(*v))
        };
        match self {
            Filter::One(part) => any(part),
            Filter::And(a, b) => any(a) && any(b),
            Filter::Or(a, b) => any(a) || any(b),
        }
    }

    fn expr(&self) -> Expr {
        match self {
            Filter::One(part) => part.expr(),
            Filter::And(a, b) => a.expr().and(b.expr()),
            Filter::Or(a, b) => a.expr().or(b.expr()),
        }
    }
}

fn number() -> impl Strategy<Value = f64> {
    (-4i32..=4).prop_map(f64::from)
}

fn part() -> impl Strategy<Value = Part> {
    prop_oneof![
        number().prop_map(Part::Eq),
        number().prop_map(Part::Lt),
        number().prop_map(Part::LtEq),
        number().prop_map(Part::Gt),
        number().prop_map(Part::GtEq),
        (number(), number()).prop_map(|(a, b)| Part::Between(a, b)),
        prop::collection::vec(number(), 1..4).prop_map(Part::In),
    ]
}

fn filter() -> impl Strategy<Value = Filter> {
    prop_oneof![
        part().prop_map(Filter::One),
        (part(), part()).prop_map(|(a, b)| Filter::And(a, b)),
        (part(), part()).prop_map(|(a, b)| Filter::Or(a, b)),
    ]
}

fn value() -> impl Strategy<Value = Option<f64>> {
    prop_oneof![
        6 => number().prop_map(Some),
        1 => Just(None),
        1 => Just(Some(f64::NAN)),
    ]
}

/// A row: the values of `v`, or `None` for a row without `v`.
fn rows() -> impl Strategy<Value = Vec<Option<Vec<Option<f64>>>>> {
    let row = prop_oneof![
        5 => prop::collection::vec(value(), 1..5).prop_map(Some),
        1 => Just(None),
    ];
    prop::collection::vec(row, 1..8)
}

fn nd_row(id: usize, values: &Option<Vec<Option<f64>>>) -> NdRecordBatch {
    let (name, array): (&str, NdArrowArray) = match values {
        Some(values) => {
            let dims = Dimensions::try_new([("i", values.len())]).unwrap();
            let array = Arc::new(Float64Array::from(values.clone()));
            ("v", NdArrowArray::try_new(array, dims).unwrap())
        }
        None => {
            let dims = Dimensions::try_new([("j", 1)]).unwrap();
            let array = Arc::new(Int32Array::from(vec![0]));
            ("w", NdArrowArray::try_new(array, dims).unwrap())
        }
    };
    let data_type = array.values().data_type().clone();
    let schema = Arc::new(Schema::new(vec![Field::new(name, data_type, true)]));
    NdRecordBatch::try_new_inferred(schema, vec![array])
        .unwrap()
        .with_attributes(Attributes::new().with("id", id as i64))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn a_stats_filter_keeps_each_matching_row(table in rows(), filter in filter()) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let kept: HashSet<i64> = runtime.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
            let options = WriteOptions { stats: true, ..Default::default() };
            let batches: Vec<_> = table.iter().enumerate().map(|(id, v)| Ok(nd_row(id, v))).collect();
            let dataset = write(&uri, stream::iter(batches), &options).await.unwrap();
            scan(&dataset)
                .filter_expr(filter.expr())
                .try_collect()
                .await
                .unwrap()
                .iter()
                .map(|row| row.attributes().get_primitive::<Int64Type>("id").unwrap())
                .collect()
        });
        for (id, values) in table.iter().enumerate() {
            let matches = values.as_ref().is_some_and(|values| filter.matches(values));
            prop_assert!(
                !matches || kept.contains(&(id as i64)),
                "row {id} {values:?} matches {filter:?} but the scan skipped it"
            );
        }
    }
}
