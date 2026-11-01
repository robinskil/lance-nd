//! Encode nd record batches as rows of a Lance table, and decode them.
//!
//! The table schema is the nd-arrow encoding: one `nd_arrow.array` column per
//! variable and an optional `__global_attrs__` column. Each row is one
//! [`NdRecordBatch`].

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch, RecordBatchOptions, StructArray};
use arrow::datatypes::{Field, FieldRef, Schema};
use nd_arrow_array::extension::{
    GLOBAL_ATTRIBUTES_FIELD, global_attributes_field, is_global_attributes_field, is_nd_field,
    logical_field,
};
use nd_arrow_array::{Attributes, NdColumn, NdColumnBuilder, NdRecordBatch};

use crate::error::{Error, Result};
use crate::options::without_encoding_metadata;
use crate::schema::conform_batch;

/// Encode `rows` as one record batch, one row per nd record batch.
///
/// The columns are the union of the variables of all rows. A row without a
/// variable has a null in its column. One variable must have one value type.
pub fn encode_rows(rows: &[NdRecordBatch]) -> Result<RecordBatch> {
    let mut variables: Vec<FieldRef> = Vec::new();
    for field in rows.iter().flat_map(|r| r.schema().fields().iter()) {
        if field.name() == GLOBAL_ATTRIBUTES_FIELD {
            return Err(Error::SchemaMismatch(format!(
                "variable name '{GLOBAL_ATTRIBUTES_FIELD}' is reserved"
            )));
        }
        match variables.iter().find(|v| v.name() == field.name()) {
            Some(existing) if existing.data_type() != field.data_type() => {
                return Err(Error::SchemaMismatch(format!(
                    "variable '{}' has value types {} and {}",
                    field.name(),
                    existing.data_type(),
                    field.data_type()
                )));
            }
            Some(_) => {}
            None => variables.push(field.clone()),
        }
    }

    let mut fields: Vec<Field> = Vec::with_capacity(variables.len() + 1);
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(variables.len() + 1);
    for variable in &variables {
        let mut builder = NdColumnBuilder::with_value_type(variable.data_type().clone());
        for row in rows {
            builder.append_option(row.column_by_name(variable.name()))?;
        }
        let column = builder.finish()?;
        let field = column.to_field(variable.name());
        // Keep the metadata of the logical field. The extension keys win.
        let mut metadata = variable.metadata().clone();
        metadata.extend(field.metadata().clone());
        fields.push(field.with_metadata(metadata));
        columns.push(column.into_array_ref());
    }

    if rows.iter().any(|r| !r.attributes().is_empty()) {
        let global = Attributes::union_fields(rows.iter().map(NdRecordBatch::attributes))?;
        let attrs: Vec<Option<&Attributes>> = rows.iter().map(|r| Some(r.attributes())).collect();
        columns.push(Arc::new(Attributes::encode_rows(&attrs, &global)?));
        // Nullable, so that Lance can add the column to an existing table.
        fields.push(global_attributes_field(global).with_nullable(true));
    }

    let options = RecordBatchOptions::new().with_row_count(Some(rows.len()));
    Ok(RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        columns,
        &options,
    )?)
}

/// The logical field of an nd column, without the Lance encoding keys.
pub fn variable_field(encoded: &Field) -> Result<Field> {
    Ok(logical_field(&without_encoding_metadata(encoded))?)
}

/// Decodes the rows of one record batch of a table, one row at a time.
///
/// A null variable is absent from its row. The grid of each row is inferred
/// from its variables, as in the nd-arrow format.
#[derive(Debug, Clone)]
pub struct RowDecoder {
    variables: Vec<(FieldRef, NdColumn)>,
    global: Option<StructArray>,
    num_rows: usize,
}

impl RowDecoder {
    /// A decoder for `batch`. Each column must be an nd column or the
    /// global attributes column. The Lance encoding keys are removed.
    pub fn try_new(batch: &RecordBatch) -> Result<Self> {
        let fields: Vec<Field> = batch
            .schema()
            .fields()
            .iter()
            .map(|field| without_encoding_metadata(field))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let batch = &conform_batch(batch, &schema)?;
        let mut variables = Vec::new();
        let mut global = None;
        for (field, column) in schema.fields().iter().zip(batch.columns()) {
            if is_global_attributes_field(field) {
                global = column.as_struct_opt().cloned();
            } else if is_nd_field(field) {
                let logical = Arc::new(logical_field(field)?);
                variables.push((logical, NdColumn::try_from_array(column.as_ref())?));
            } else {
                return Err(Error::InvalidTable(format!(
                    "column '{}' is not an nd column",
                    field.name()
                )));
            }
        }
        Ok(Self {
            variables,
            global,
            num_rows: batch.num_rows(),
        })
    }

    /// The number of rows in the batch.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Decode row `row`. Panics when `row` is out of bounds.
    pub fn row(&self, row: usize) -> Result<NdRecordBatch> {
        let mut fields = Vec::new();
        let mut arrays = Vec::new();
        for (field, column) in &self.variables {
            if !column.is_null(row) {
                fields.push(field.clone());
                arrays.push(column.value(row));
            }
        }
        let mut batch = NdRecordBatch::try_new_inferred(Arc::new(Schema::new(fields)), arrays)?;
        if let Some(global) = self.global.as_ref().filter(|g| g.is_valid(row)) {
            batch = batch.with_attributes(Attributes::from_struct_row(global, row));
        }
        Ok(batch)
    }

    /// Decode the rows one at a time, in order.
    pub fn into_rows(self) -> impl Iterator<Item = Result<NdRecordBatch>> + Send + 'static {
        (0..self.num_rows).map(move |row| self.row(row))
    }
}
