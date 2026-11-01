//! Merge table schemas and shape record batches to a table schema.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, ListArray, RecordBatch, StructArray, new_empty_array, new_null_array,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
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
    Ok(RecordBatch::try_new(schema.clone(), columns)?)
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
