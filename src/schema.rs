//! Merge table schemas and shape record batches to a table schema.

use std::sync::Arc;

use arrow::array::BooleanBufferBuilder;
use arrow::array::{
    Array, ArrayRef, AsArray, ListArray, RecordBatch, RecordBatchOptions, StructArray, make_array,
    new_empty_array, new_null_array,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::compute::{concat, is_null, kernels::zip::zip};
use arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};

use crate::error::{Error, Result};

/// The union of two schemas. Fields of `base` keep their order, nullability
/// and metadata. Struct fields merge their children by name.
pub fn merge_schemas(base: &Schema, other: &Schema) -> Result<Schema> {
    let fields = merge_children(base.fields(), other.fields(), "")?;
    Ok(Schema::new_with_metadata(fields, base.metadata().clone()))
}

fn merge_children(base: &Fields, other: &Fields, parent: &str) -> Result<Fields> {
    let mut fields = base
        .iter()
        .map(|field| match other.find(field.name()) {
            Some((_, o)) => {
                merge_field(field, o, &format!("{parent}{}", field.name())).map(Arc::new)
            }
            None => Ok(field.clone()),
        })
        .collect::<Result<Vec<FieldRef>>>()?;
    fields.extend(
        other
            .iter()
            .filter(|f| base.find(f.name()).is_none())
            .cloned(),
    );
    Ok(fields.into())
}

/// Merge two fields with the same name. `path` is the dotted name for errors.
fn merge_field(base: &Field, other: &Field, path: &str) -> Result<Field> {
    let data_type = match (base.data_type(), other.data_type()) {
        (a, b) if a == b => return Ok(base.clone()),
        (DataType::Struct(a), DataType::Struct(b)) => {
            DataType::Struct(merge_children(a, b, &format!("{path}."))?)
        }
        (DataType::List(a), DataType::List(b)) => {
            DataType::List(Arc::new(merge_field(a, b, &format!("{path}[]"))?))
        }
        (a, b) => {
            return Err(Error::SchemaMismatch(format!(
                "field '{path}' has type {a} in the table and {b} in the new rows"
            )));
        }
    };
    Ok(base.clone().with_data_type(data_type))
}

/// The fields of `merged` that `existing` lacks. A struct that exists but
/// lacks children gives a struct with only the missing children.
pub fn missing_fields(existing: &Fields, merged: &Fields) -> Vec<FieldRef> {
    merged
        .iter()
        .filter_map(|field| {
            let Some((_, current)) = existing.find(field.name()) else {
                return Some(field.clone());
            };
            match (current.data_type(), field.data_type()) {
                (DataType::Struct(current), DataType::Struct(children)) => {
                    let missing = missing_fields(current, children);
                    (!missing.is_empty()).then(|| {
                        Arc::new(
                            field
                                .as_ref()
                                .clone()
                                .with_data_type(DataType::Struct(missing.into())),
                        )
                    })
                }
                _ => None,
            }
        })
        .collect()
}

/// Shape `batch` to `schema`: reorder fields by name and fill missing
/// fields with [`absent`] values, also inside structs.
pub fn conform_batch(batch: &RecordBatch, schema: &SchemaRef) -> Result<RecordBatch> {
    let columns = schema
        .fields()
        .iter()
        .map(|field| match batch.column_by_name(field.name()) {
            Some(column) => conform(column, field),
            None => absent(field, batch.num_rows()),
        })
        .collect::<Result<Vec<_>>>()?;
    let options = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
    Ok(RecordBatch::try_new_with_options(
        schema.clone(),
        columns,
        &options,
    )?)
}

/// The rows of `batches` as one batch of `schema`. A column or a struct
/// child that a batch lacks gets absent rows, as [`conform_batch`] gives
/// them.
pub fn concat_conformed(schema: &SchemaRef, batches: &[RecordBatch]) -> Result<RecordBatch> {
    let rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    let columns = schema
        .fields()
        .iter()
        .map(|field| {
            let parts: Vec<Part> = batches
                .iter()
                .map(|batch| {
                    (
                        batch.column_by_name(field.name()).cloned(),
                        batch.num_rows(),
                    )
                })
                .collect();
            concat_parts(field, &parts)
        })
        .collect::<Result<Vec<_>>>()?;
    let options = RecordBatchOptions::new().with_row_count(Some(rows));
    Ok(RecordBatch::try_new_with_options(
        schema.clone(),
        columns,
        &options,
    )?)
}

/// The array of one batch, or `None` when the batch lacks it, and the rows.
type Part = (Option<ArrayRef>, usize);

/// One array of `field` from `parts`. A struct is built child by child, so
/// a batch with few of many children costs little.
fn concat_parts(field: &Field, parts: &[Part]) -> Result<ArrayRef> {
    if let DataType::Struct(children) = field.data_type()
        && !children.is_empty()
    {
        let columns = children
            .iter()
            .map(|child| {
                let child_parts: Vec<Part> = parts
                    .iter()
                    .map(|(array, rows)| {
                        let column = array
                            .as_ref()
                            .and_then(|array| array.as_struct().column_by_name(child.name()));
                        (column.cloned(), *rows)
                    })
                    .collect();
                concat_parts(child, &child_parts)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut validity = BooleanBufferBuilder::new(parts.iter().map(|(_, rows)| rows).sum());
        for (array, rows) in parts {
            match array {
                Some(array) => match array.logical_nulls() {
                    Some(nulls) => validity.append_buffer(nulls.inner()),
                    None => validity.append_n(*rows, true),
                },
                // As `absent`: a nullable struct is null, else valid.
                None => validity.append_n(*rows, !field.is_nullable()),
            }
        }
        let nulls = NullBuffer::new(validity.finish());
        let nulls = (nulls.null_count() > 0).then_some(nulls);
        return Ok(Arc::new(StructArray::try_new(
            children.clone(),
            columns,
            nulls,
        )?));
    }
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(parts.len());
    let mut missing = 0;
    for (array, rows) in parts {
        match array {
            Some(array) => {
                // One absent array for each run of batches without the field.
                if missing > 0 {
                    arrays.push(absent(field, missing)?);
                    missing = 0;
                }
                arrays.push(conform(array, field)?);
            }
            None => missing += rows,
        }
    }
    if missing > 0 || arrays.is_empty() {
        arrays.push(absent(field, missing)?);
    }
    let arrays: Vec<&dyn Array> = arrays.iter().map(|array| array.as_ref()).collect();
    Ok(match arrays.as_slice() {
        [one] => make_array(one.to_data()),
        _ => concat(&arrays)?,
    })
}

/// The top-level fields of `staged` that `schemas` use, in the order of
/// `staged`. A struct keeps all its children: Lance 11 cannot read a
/// fragment whose struct has fewer children than the table.
pub fn used_fields(staged: &Schema, schemas: &[SchemaRef]) -> Schema {
    let fields: Vec<FieldRef> = staged
        .fields()
        .iter()
        .filter(|field| {
            schemas
                .iter()
                .any(|schema| schema.column_with_name(field.name()).is_some())
        })
        .cloned()
        .collect();
    Schema::new_with_metadata(fields, staged.metadata().clone())
}

/// `true` when `staged` has each field of `new` with the same type. A
/// struct of `new` can have fewer children.
pub fn holds_schema(staged: &Schema, new: &Schema) -> bool {
    new.fields().iter().all(|field| {
        staged
            .field_with_name(field.name())
            .is_ok_and(|staged| holds(staged.data_type(), field.data_type()))
    })
}

fn holds(staged: &DataType, new: &DataType) -> bool {
    match (staged, new) {
        (DataType::Struct(staged), DataType::Struct(new)) => new.iter().all(|field| {
            staged
                .find(field.name())
                .is_some_and(|(_, staged)| holds(staged.data_type(), field.data_type()))
        }),
        (DataType::List(staged), DataType::List(new)) => holds(staged.data_type(), new.data_type()),
        (staged, new) => staged == new,
    }
}

fn conform(array: &ArrayRef, target: &Field) -> Result<ArrayRef> {
    if array.data_type() == target.data_type() {
        return Ok(array.clone());
    }
    match (array.data_type(), target.data_type()) {
        (DataType::Struct(_), DataType::Struct(fields)) => {
            let source = array.as_struct();
            let children = fields
                .iter()
                .map(|field| match source.column_by_name(field.name()) {
                    Some(child) => conform(child, field),
                    None => absent(field, source.len()),
                })
                .collect::<Result<Vec<_>>>()?;
            let nulls = source.nulls().cloned();
            Ok(Arc::new(StructArray::try_new(
                fields.clone(),
                children,
                nulls,
            )?))
        }
        (DataType::List(_), DataType::List(item)) => {
            let source = array.as_list::<i32>();
            let values = conform(source.values(), item)?;
            let offsets = source.offsets().clone();
            let nulls = source.nulls().cloned();
            Ok(Arc::new(ListArray::try_new(
                item.clone(),
                offsets,
                values,
                nulls,
            )?))
        }
        (from, to) => Err(Error::SchemaMismatch(format!(
            "field '{}' has type {from} in the new rows and {to} in the table",
            target.name()
        ))),
    }
}

/// `array` without nulls in the non-nullable fields of `field`.
///
/// Lance reads the children of a null struct row as null, but rejects nulls
/// in a non-nullable field in a write. Each such null becomes the value of
/// [`absent`], as a writer stores it.
pub fn fill_hidden_nulls(array: &ArrayRef, field: &Field) -> Result<ArrayRef> {
    let array = if !field.is_nullable() && array.null_count() > 0 {
        let mask = is_null(array.as_ref())?;
        zip(&mask, &absent(field, array.len())?, array)?
    } else {
        array.clone()
    };
    match field.data_type() {
        DataType::Struct(fields) if !fields.is_empty() => {
            let source = array.as_struct();
            let children = fields
                .iter()
                .zip(source.columns())
                .map(|(field, child)| fill_hidden_nulls(child, field))
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(StructArray::try_new(
                fields.clone(),
                children,
                source.nulls().cloned(),
            )?))
        }
        DataType::List(item) => {
            let source = array.as_list::<i32>();
            Ok(Arc::new(ListArray::try_new(
                item.clone(),
                source.offsets().clone(),
                fill_hidden_nulls(source.values(), item)?,
                source.nulls().cloned(),
            )?))
        }
        _ => Ok(array),
    }
}

/// `len` values for a field that the new rows lack.
///
/// A nullable field gets nulls. A non-nullable struct gets valid rows of
/// absent children, and a non-nullable list gets empty lists. Children of a
/// null struct stay valid, because Lance rejects nulls in non-nullable fields.
/// Thus a missing nd variable gives null rows with empty value lists.
pub fn absent(field: &Field, len: usize) -> Result<ArrayRef> {
    let nulls = field.is_nullable().then(|| NullBuffer::new_null(len));
    match field.data_type() {
        DataType::Struct(fields) => {
            let children = fields
                .iter()
                .map(|child| absent(child, len))
                .collect::<Result<Vec<_>>>()?;
            if fields.is_empty() {
                return Ok(Arc::new(StructArray::new_empty_fields(len, nulls)));
            }
            Ok(Arc::new(StructArray::try_new(
                fields.clone(),
                children,
                nulls,
            )?))
        }
        DataType::List(item) => Ok(Arc::new(ListArray::try_new(
            item.clone(),
            OffsetBuffer::new_zeroed(len),
            new_empty_array(item.data_type()),
            nulls,
        )?)),
        _ if field.is_nullable() => Ok(new_null_array(field.data_type(), len)),
        data_type => Err(Error::SchemaMismatch(format!(
            "the new rows lack the required field '{}' of type {data_type}",
            field.name()
        ))),
    }
}
