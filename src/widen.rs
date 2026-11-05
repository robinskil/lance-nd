//! Widen the value type of a variable in the table.
//!
//! Lance 11 cannot cast a struct column or a nested field in place, so
//! [`widen`] writes the variable again under new field ids and commits it in
//! place of the old column. The stats column gets the same treatment. Only
//! these columns are written again, not the whole table.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef};
use lance::Dataset;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{BatchUDF, CommitBuilder, NewColumnTransform};
use lance_core::datatypes::{Field as LanceField, format_field_path};
use nd_arrow_array::extension::{VALUES_FIELD, is_nd_field};

use crate::cast::{Casting, allowed, promote_type};
use crate::error::{Error, Result};
use crate::schema::fill_hidden_nulls;
use crate::stats::stats_column;

/// The name prefix of the new columns until the commit gives them the old names.
const TEMPORARY_PREFIX: &str = "__widen__";

/// Change the value type of `variable` to `to`, for example `Float32` to
/// `Float64`. The stats of the variable get the new type too.
///
/// `to` must hold each old value exactly, as [`Casting::Safe`] allows. The
/// same type does nothing. A widen is one Lance commit. It writes the
/// variable again, so a concurrent commit that adds rows makes it start
/// again, up to a limit.
pub async fn widen(dataset: &mut Dataset, variable: &str, to: &DataType) -> Result<()> {
    let schema = Schema::from(dataset.schema());
    let field = schema
        .field_with_name(variable)
        .ok()
        .filter(|field| is_nd_field(field))
        .ok_or_else(|| {
            Error::InvalidArgument(format!("'{variable}' is not a variable of the table"))
        })?;
    let from = value_type(field)?;
    if &from == to {
        return Ok(());
    }
    if !allowed(&from, to, Casting::Safe) {
        return Err(Error::InvalidArgument(format!(
            "cannot widen '{variable}' from {from} to {to}: some values could change"
        )));
    }
    let mut columns = vec![(field.clone(), with_value_type(field, to))];
    if let Ok(stats) = schema.field_with_name(&stats_column(variable)) {
        columns.push((stats.clone(), map_children(stats, |_| to.clone())));
    }
    rewrite(dataset, &columns).await
}

/// The variables of `batch` that need a wider type in `table` under
/// [`Casting::Promote`], with that type.
pub(crate) fn promotions(batch: &RecordBatch, table: &Schema) -> Vec<(String, DataType)> {
    batch
        .schema()
        .fields()
        .iter()
        .filter(|field| is_nd_field(field))
        .filter_map(|field| {
            let staged = table.field_with_name(field.name()).ok()?;
            let (new, old) = (value_type(field).ok()?, value_type(staged).ok()?);
            if allowed(&new, &old, Casting::Safe) {
                return None;
            }
            let to = promote_type(&old, &new)?;
            (to != old).then(|| (field.name().clone(), to))
        })
        .collect()
}

/// The value type of an nd column.
fn value_type(field: &Field) -> Result<DataType> {
    if let DataType::Struct(children) = field.data_type()
        && let Some((_, values)) = children.find(VALUES_FIELD)
        && let DataType::List(item) = values.data_type()
    {
        return Ok(item.data_type().clone());
    }
    Err(Error::InvalidTable(format!(
        "column '{}' is not an nd column",
        field.name()
    )))
}

/// `field` with `data_type` for each child, as `retype` gives it.
fn map_children(field: &Field, retype: impl Fn(&Field) -> DataType) -> Field {
    let DataType::Struct(children) = field.data_type() else {
        return field.clone();
    };
    let children: Fields = children
        .iter()
        .map(|child| Arc::new(child.as_ref().clone().with_data_type(retype(child))))
        .collect();
    field.clone().with_data_type(DataType::Struct(children))
}

/// An nd column with values of type `to`.
fn with_value_type(field: &Field, to: &DataType) -> Field {
    map_children(field, |child| match child.data_type() {
        DataType::List(item) if child.name() == VALUES_FIELD => {
            DataType::List(Arc::new(item.as_ref().clone().with_data_type(to.clone())))
        }
        other => other.clone(),
    })
}

/// The most tries of a widen that a concurrent commit preempts.
const MAX_ATTEMPTS: usize = 10;

/// Write each old column of `columns` again with the type of its new field,
/// in one Lance commit.
///
/// Each fragment gets a new data file with the new columns, under new field
/// ids and the old names. A concurrent append or writer commit conflicts with
/// this commit, so no row keeps a field id that the widen removes.
async fn rewrite(dataset: &mut Dataset, columns: &[(Field, Field)]) -> Result<()> {
    for attempt in 1..=MAX_ATTEMPTS {
        match rewrite_once(dataset, columns).await {
            Err(Error::Lance(error))
                if matches!(*error, lance::Error::RetryableCommitConflict { .. })
                    && attempt < MAX_ATTEMPTS =>
            {
                dataset.checkout_latest().await?;
            }
            result => return result,
        }
    }
    unreachable!("the last attempt returns")
}

async fn rewrite_once(dataset: &mut Dataset, columns: &[(Field, Field)]) -> Result<()> {
    let temporary: Vec<Field> = columns
        .iter()
        .map(|(_, new)| {
            let name = format!("{TEMPORARY_PREFIX}{}", new.name());
            new.clone().with_name(name)
        })
        .collect();
    let output = Arc::new(Schema::new(temporary));
    let old: Vec<String> = columns.iter().map(|(old, _)| old.name().clone()).collect();
    let paths: Vec<String> = old.iter().map(|name| format_field_path(&[name])).collect();
    let read_version = dataset.version().version;

    let mut fragments = Vec::new();
    let mut schema = None;
    for fragment in dataset.get_fragments() {
        let udf = NewColumnTransform::BatchUDF(cast_udf(output.clone(), old.clone()));
        let (fragment, fragment_schema) =
            fragment.add_columns(udf, Some(paths.clone()), None).await?;
        fragments.push(fragment);
        schema = Some(fragment_schema);
    }
    // Without fragments, only the schema changes. Lance gives new fields the
    // ids after the highest id, as add_columns does.
    let mut schema = match schema {
        Some(schema) => schema,
        None => {
            let mut schema = dataset.schema().merge(output.as_ref())?;
            schema.set_field_id(Some(dataset.manifest.max_field_id()));
            schema
        }
    };
    // Each old leaf takes the id and the type of its new leaf. A data file
    // lists leaf ids only, so the structs and lists keep their ids. Rows with
    // the old leaf ids no longer read, also from a concurrent writer.
    for name in &old {
        let temporary = schema
            .field(&format!("{TEMPORARY_PREFIX}{name}"))
            .cloned()
            .ok_or_else(|| Error::InvalidTable(format!("no new column for '{name}'")))?;
        let field = schema
            .fields
            .iter_mut()
            .find(|field| field.name == *name)
            .ok_or_else(|| Error::InvalidTable(format!("no column '{name}'")))?;
        take_leaves(field, &temporary);
    }
    schema
        .fields
        .retain(|field| !field.name.starts_with(TEMPORARY_PREFIX));
    // A data file with only old fields holds nothing that the table reads.
    let ids: HashSet<i32> = schema.field_ids().into_iter().collect();
    for fragment in &mut fragments {
        fragment
            .files
            .retain(|file| file.fields.iter().any(|id| ids.contains(id)));
    }
    let operation = Operation::Merge {
        fragments,
        schema,
        preserves_nullability: true,
    };
    *dataset = CommitBuilder::new(Arc::new(dataset.clone()))
        .execute(Transaction::new(read_version, operation, None))
        .await?;
    Ok(())
}

/// Give each leaf of `field` the leaf at the same path in `new`.
fn take_leaves(field: &mut LanceField, new: &LanceField) {
    if !field.data_type().is_nested() {
        let parent_id = field.parent_id;
        *field = new.clone();
        field.parent_id = parent_id;
        return;
    }
    for child in &mut field.children {
        if let Some(new_child) = new.children.iter().find(|c| c.name == child.name) {
            take_leaves(child, new_child);
        }
    }
}

/// A transform that reads the columns `old` and casts them to the fields of
/// `output`.
fn cast_udf(output: SchemaRef, old: Vec<String>) -> BatchUDF {
    let mapper_output = output.clone();
    BatchUDF {
        mapper: Box::new(move |batch| {
            let arrays = mapper_output
                .fields()
                .iter()
                .zip(&old)
                .map(|(field, name)| {
                    let column = batch.column_by_name(name).ok_or_else(|| {
                        lance::Error::invalid_input(format!("no column '{name}' to widen"))
                    })?;
                    let column = cast(column, field.data_type())?;
                    // Lance reads the children of an absent variable as null,
                    // but rejects them in a write.
                    fill_hidden_nulls(&column, field)
                        .map_err(|error| lance::Error::invalid_input(error.to_string()))
                })
                .collect::<lance::Result<Vec<_>>>()?;
            Ok(RecordBatch::try_new(mapper_output.clone(), arrays)?)
        }),
        output_schema: output,
        result_checkpoint: None,
    }
}
