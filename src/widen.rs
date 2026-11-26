//! Widen the value type of a variable in the table.
//!
//! Lance 11 cannot cast a struct column or a nested field in place, so
//! [`widen`] writes the variable again under new field ids and commits it in
//! place of the old column. The stats column gets the same treatment. Only
//! these columns are written again, not the whole table.
//!
//! No widen commits a Lance Merge through `CommitBuilder`: its validation
//! takes a time that grows with the square of the schema fields.

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef};
use futures::FutureExt;
use lance::Dataset;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{BatchUDF, CommitBuilder, NewColumnTransform};
use lance_core::datatypes::{Field as LanceField, Schema as LanceSchema, format_field_path};
use nd_arrow_array::extension::{VALUES_FIELD, is_nd_field};

use crate::cast::{Casting, allowed, promote_type};
use crate::error::{Error, Result};
use crate::format::check_format;
use crate::schema::fill_hidden_nulls;
use crate::stats::stats_column;

/// The name prefix of the new columns until the commit gives them the old
/// names. It starts with the reserved stats prefix, so scans skip these columns.
const TEMPORARY_PREFIX: &str = "__stats__widen__";

/// Change the value type of `variable` to `to`, for example `Float32` to
/// `Float64`. The stats of the variable get the new type too.
///
/// `to` must hold each old value exactly, as [`Casting::Safe`] allows. The
/// same type does nothing. A widen writes the variable again in a new
/// column, then swaps it in with a second Lance commit. Readers see the old
/// or the new type, never a mix. A concurrent commit that adds rows makes it
/// start again, up to a limit.
#[tracing::instrument(skip_all, fields(uri = %dataset.uri(), variable = %variable, to = %to), err)]
pub async fn widen(dataset: &mut Dataset, variable: &str, to: &DataType) -> Result<()> {
    // Boxed: the rewrite future is large, and exceeds the layout depth of callers.
    widen_all(dataset, &[(variable.to_string(), to.clone())], false)
        .boxed()
        .await
}

/// Change the value types of several variables as [`widen`] does, in one
/// rewrite. A `detached` widen makes detached versions.
pub(crate) async fn widen_all(
    dataset: &mut Dataset,
    widenings: &[(String, DataType)],
    detached: bool,
) -> Result<()> {
    check_format(dataset)?;
    let schema = Schema::from(dataset.schema());
    let mut columns = Vec::new();
    for (variable, to) in widenings {
        columns.extend(widen_columns(&schema, variable, to)?);
    }
    if columns.is_empty() {
        return Ok(());
    }
    rewrite(dataset, &columns, detached).await
}

/// The old and new fields of `variable` and its stats for a widen to `to`.
/// Empty when `to` is the type of `variable`.
fn widen_columns(schema: &Schema, variable: &str, to: &DataType) -> Result<Vec<(Field, Field)>> {
    let field = schema
        .field_with_name(variable)
        .ok()
        .filter(|field| is_nd_field(field))
        .ok_or_else(|| {
            Error::InvalidArgument(format!("'{variable}' is not a variable of the table"))
        })?;
    let from = value_type(field)?;
    if &from == to {
        return Ok(Vec::new());
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
    Ok(columns)
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

/// Write each old column of `columns` again with the type of its new field.
///
/// Each fragment gets a new data file with the new columns, under new field
/// ids. A schema commit then gives them the old names. A concurrent append or
/// writer commit conflicts with both commits, so no row keeps a field id that
/// the widen removes. A failed widen drops its temporary columns.
async fn rewrite(dataset: &mut Dataset, columns: &[(Field, Field)], detached: bool) -> Result<()> {
    for attempt in 1..=MAX_ATTEMPTS {
        match rewrite_once(dataset, columns, detached).await {
            Err(Error::CommitConflict {
                retryable: true, ..
            }) if attempt < MAX_ATTEMPTS && !detached => {
                tracing::debug!(
                    attempt,
                    "a concurrent commit preempted the widen, try again"
                );
                dataset.checkout_latest().await?;
            }
            Err(error) if !detached => {
                drop_leftovers(dataset).await;
                return Err(error);
            }
            result => return result,
        }
    }
    unreachable!("the last attempt returns")
}

/// Drop the temporary columns at the latest version of the table of
/// `dataset`, and keep `dataset` as it is. A failure only logs a warning:
/// readers do not see these columns, and the next widen drops them.
async fn drop_leftovers(dataset: &Dataset) {
    let result = async {
        let mut latest = dataset.clone();
        latest.checkout_latest().await?;
        drop_temporary(&mut latest).await
    }
    .await;
    if let Err(error) = result {
        tracing::warn!(%error, "cannot drop the temporary columns of a failed widen");
    }
}

async fn rewrite_once(
    dataset: &mut Dataset,
    columns: &[(Field, Field)],
    detached: bool,
) -> Result<()> {
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
    let udf = || NewColumnTransform::BatchUDF(cast_udf(output.clone(), old.clone()));

    if !detached {
        drop_temporary(dataset).await?;
        // A Merge that skips the quadratic validation of CommitBuilder.
        dataset
            .add_columns(udf(), Some(paths), None)
            .boxed()
            .await?;
        let schema = swap_leaves(dataset.schema().clone(), &old)?;
        return commit_schema(dataset, schema, false).await;
    }

    // No concurrent commits on a detached version, so an Update adds the files.
    let mut fragments = Vec::new();
    let mut schema = None;
    for fragment in dataset.get_fragments() {
        let (fragment, fragment_schema) = fragment
            .add_columns(udf(), Some(paths.clone()), None)
            .await?;
        fragments.push(fragment);
        schema = Some(fragment_schema);
    }
    // Without fragments, only the schema changes. Lance gives new fields the
    // ids after the highest id, as add_columns does.
    let schema = match schema {
        Some(schema) => schema,
        None => {
            let mut schema = dataset.schema().merge(output.as_ref())?;
            schema.set_field_id(Some(dataset.manifest.max_field_id()));
            schema
        }
    };
    let schema = swap_leaves(schema, &old)?;
    if !fragments.is_empty() {
        let operation = Operation::Update {
            removed_fragment_ids: Vec::new(),
            updated_fragments: fragments,
            new_fragments: Vec::new(),
            fields_modified: Vec::new(),
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: Vec::new(),
            update_mode: None,
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        };
        *dataset = CommitBuilder::new(Arc::new(dataset.clone()))
            .with_detached(true)
            .execute(Transaction::new(dataset.version().version, operation, None))
            .await?;
    }
    commit_schema(dataset, schema, true).await
}

/// Drop the temporary columns of a widen that did not finish.
async fn drop_temporary(dataset: &mut Dataset) -> Result<()> {
    let names: Vec<String> = dataset
        .schema()
        .fields
        .iter()
        .filter(|field| field.name.starts_with(TEMPORARY_PREFIX))
        .map(|field| field.name.clone())
        .collect();
    if names.is_empty() {
        return Ok(());
    }
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    dataset.drop_columns(&names).boxed().await?;
    Ok(())
}

/// Commit `schema` as the table schema with a Lance Project, whose
/// validation takes a time linear in the schema fields. Lance drops the data
/// files that hold no field of `schema`.
///
/// Without `preserves_nullability`, the Project conflicts with each
/// concurrent commit that adds rows, in both orders. Those rows can hold
/// field ids that `schema` drops.
pub(crate) async fn commit_schema(
    dataset: &mut Dataset,
    schema: LanceSchema,
    detached: bool,
) -> Result<()> {
    let operation = Operation::Project {
        schema,
        preserves_nullability: false,
    };
    *dataset = CommitBuilder::new(Arc::new(dataset.clone()))
        .with_detached(detached)
        .execute(Transaction::new(dataset.version().version, operation, None))
        .await?;
    Ok(())
}

/// `schema` without its temporary columns, where each column of `old` takes
/// the leaves of its temporary column.
fn swap_leaves(mut schema: LanceSchema, old: &[String]) -> Result<LanceSchema> {
    // Each old leaf takes the id and the type of its new leaf. A data file
    // lists leaf ids only, so the structs and lists keep their ids. Rows with
    // the old leaf ids no longer read, also from a concurrent writer.
    for name in old {
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
    Ok(schema)
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

#[cfg(test)]
mod tests {
    use arrow::array::Float32Array;
    use futures::stream;
    use nd_arrow_array::{Dimensions, NdArrowArray, NdRecordBatch};

    use super::*;
    use crate::write::{WriteOptions, write};

    #[tokio::test]
    async fn drop_leftovers_drops_the_temporary_columns_of_the_latest_version() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
        let field = Field::new("v", DataType::Float32, true);
        let array = NdArrowArray::try_new(
            Arc::new(Float32Array::from(vec![1.5f32])),
            Dimensions::try_new([("x", 1)]).unwrap(),
        )
        .unwrap();
        let row = NdRecordBatch::try_new_inferred(Arc::new(Schema::new(vec![field])), vec![array])
            .unwrap();
        let mut dataset = write(&uri, stream::iter([Ok(row)]), &WriteOptions::default())
            .await
            .unwrap();
        let stale = dataset.clone();
        let leftover = Schema::new(vec![Field::new(
            format!("{TEMPORARY_PREFIX}v"),
            DataType::Int32,
            true,
        )]);
        dataset
            .add_columns(NewColumnTransform::AllNulls(Arc::new(leftover)), None, None)
            .await
            .unwrap();

        drop_leftovers(&stale).await;
        assert_eq!(stale.version().version, 1);
        let latest = Dataset::open(&uri).await.unwrap();
        let names: Vec<&str> = latest
            .schema()
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect();
        assert_eq!(names, ["v"]);
    }
}
