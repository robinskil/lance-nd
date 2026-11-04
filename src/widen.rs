//! Widen the value type of a variable in the table.
//!
//! Lance 11 cannot cast a struct column or a nested field in place, so
//! [`widen`] writes the variable again as a new column, drops the old
//! column and renames the new one. The stats column gets the same
//! treatment. Only these columns are written again, not the whole table.

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Fields, Schema};
use lance::Dataset;
use lance::dataset::{BatchUDF, ColumnAlteration, NewColumnTransform};
use lance_core::datatypes::format_field_path;
use nd_arrow_array::extension::{VALUES_FIELD, is_nd_field};

use crate::cast::{Casting, allowed, promote_type};
use crate::error::{Error, Result};
use crate::schema::fill_hidden_nulls;
use crate::stats::STATS_FIELD;

/// The name prefix of the temporary columns of [`widen`].
const TEMPORARY_PREFIX: &str = "__widen__";

/// Change the value type of `variable` to `to`, for example `Float32` to
/// `Float64`. The stats of the variable get the new type too.
///
/// `to` must hold each old value exactly, as [`Casting::Safe`] allows. The
/// same type does nothing. A widen takes three Lance commits. When it stops
/// between them, the next call to `widen` finishes or undoes it first.
pub async fn widen(dataset: &mut Dataset, variable: &str, to: &DataType) -> Result<()> {
    finish_interrupted(dataset).await?;
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
    if let Ok(stats) = schema.field_with_name(STATS_FIELD)
        && let DataType::Struct(children) = stats.data_type()
        && children.find(variable).is_some()
    {
        columns.push((stats.clone(), with_stats_type(stats, variable, to)));
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

/// Widen the variables of `batch` in the table as [`Casting::Promote`]
/// needs. Without that rule, nothing changes.
pub(crate) async fn promote_table(
    dataset: &mut Dataset,
    batch: &RecordBatch,
    casting: Casting,
) -> Result<()> {
    if casting != Casting::Promote {
        return Ok(());
    }
    for (variable, to) in promotions(batch, &Schema::from(dataset.schema())) {
        widen(dataset, &variable, &to).await?;
    }
    Ok(())
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

/// The stats column with the bounds of `variable` of type `to`.
fn with_stats_type(stats: &Field, variable: &str, to: &DataType) -> Field {
    map_children(stats, |child| {
        if child.name() == variable {
            map_children(child, |_| to.clone()).data_type().clone()
        } else {
            child.data_type().clone()
        }
    })
}

/// Write each old column of `columns` again with the type of its new field.
async fn rewrite(dataset: &mut Dataset, columns: &[(Field, Field)]) -> Result<()> {
    let temporary: Vec<Field> = columns
        .iter()
        .map(|(_, new)| {
            let name = format!("{TEMPORARY_PREFIX}{}", new.name());
            new.clone().with_name(name)
        })
        .collect();
    let output = Arc::new(Schema::new(temporary));
    let old: Vec<String> = columns.iter().map(|(old, _)| old.name().clone()).collect();
    let mapper_output = output.clone();
    let mapper_old = old.clone();
    let udf = BatchUDF {
        mapper: Box::new(move |batch| {
            let arrays = mapper_output
                .fields()
                .iter()
                .zip(&mapper_old)
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
    };
    let paths: Vec<String> = old.iter().map(|name| format_field_path(&[name])).collect();
    dataset
        .add_columns(NewColumnTransform::BatchUDF(udf), Some(paths.clone()), None)
        .await?;
    let paths: Vec<&str> = paths.iter().map(String::as_str).collect();
    dataset.drop_columns(&paths).await?;
    rename_temporary(dataset).await
}

/// Finish or undo a widen that stopped between its commits.
async fn finish_interrupted(dataset: &mut Dataset) -> Result<()> {
    let schema = Schema::from(dataset.schema());
    // A temporary column next to its original: the widen stopped before the
    // drop, so the original is complete.
    let stale: Vec<String> = temporary_columns(&schema)
        .filter(|(_, original)| schema.field_with_name(original).is_ok())
        .map(|(temporary, _)| format_field_path(&[temporary]))
        .collect();
    if !stale.is_empty() {
        let stale: Vec<&str> = stale.iter().map(String::as_str).collect();
        dataset.drop_columns(&stale).await?;
    }
    rename_temporary(dataset).await
}

/// Rename each temporary column to its original name.
async fn rename_temporary(dataset: &mut Dataset) -> Result<()> {
    let schema = Schema::from(dataset.schema());
    let renames: Vec<ColumnAlteration> = temporary_columns(&schema)
        .map(|(temporary, original)| {
            ColumnAlteration::new(format_field_path(&[temporary])).rename(original.to_string())
        })
        .collect();
    if !renames.is_empty() {
        dataset.alter_columns(&renames).await?;
    }
    Ok(())
}

/// The temporary columns of `schema`, with their original names.
fn temporary_columns(schema: &Schema) -> impl Iterator<Item = (&str, &str)> {
    schema.fields().iter().filter_map(|field| {
        let name = field.name().as_str();
        name.strip_prefix(TEMPORARY_PREFIX)
            .map(|original| (name, original))
    })
}
