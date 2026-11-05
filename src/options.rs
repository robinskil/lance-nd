//! Lance encoding options for the columns of a table.
//!
//! Lance reads the encoding of a column from Arrow field metadata. The
//! structural encoding is on the top-level field. Compression, byte stream
//! split and run-length encoding are on each leaf field. Lance stores this
//! metadata in the table schema, so it applies to all later appends.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use lance::Dataset;

use crate::error::{Error, Result};
use crate::format::check_format;

// The keys of `lance_encoding::constants`.
const COMPRESSION_KEY: &str = "lance-encoding:compression";
const COMPRESSION_LEVEL_KEY: &str = "lance-encoding:compression-level";
const BYTE_STREAM_SPLIT_KEY: &str = "lance-encoding:bss";
const RLE_THRESHOLD_KEY: &str = "lance-encoding:rle-threshold";
const STRUCTURAL_ENCODING_KEY: &str = "lance-encoding:structural-encoding";
const KEYS: [&str; 5] = [
    COMPRESSION_KEY,
    COMPRESSION_LEVEL_KEY,
    BYTE_STREAM_SPLIT_KEY,
    RLE_THRESHOLD_KEY,
    STRUCTURAL_ENCODING_KEY,
];

/// The prefix of Lance encoding keys in field metadata.
pub const LANCE_ENCODING_PREFIX: &str = "lance-encoding:";

/// The prefix of the table schema metadata keys that hold the column defaults.
pub const COLUMN_DEFAULTS_PREFIX: &str = "lance_nd.column_defaults.";

/// The general compression of the values of a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// No general compression.
    None,
    /// LZ4 compression.
    Lz4,
    /// Zstandard compression. Lance selects the level when `level` is `None`.
    Zstd {
        /// The Zstandard compression level.
        level: Option<i32>,
    },
}

/// The page layout of a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralEncoding {
    /// Small blocks of values. Lance prefers it for narrow values.
    MiniBlock,
    /// Values zipped with their structure. Lance prefers it for wide values.
    FullZip,
}

/// Byte stream split of floating point values before compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteStreamSplit {
    /// Never split.
    Off,
    /// Always split floating point values.
    On,
    /// Lance decides from the data.
    Auto,
}

/// Lance encoding options of one column. `None` leaves the choice to Lance.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ColumnOptions {
    /// The general compression of the values.
    pub compression: Option<Compression>,
    /// Byte stream split of floating point values.
    pub byte_stream_split: Option<ByteStreamSplit>,
    /// Lance uses run-length encoding when `runs < values * rle_threshold`.
    pub rle_threshold: Option<f64>,
    /// The page layout of the column.
    pub structural_encoding: Option<StructuralEncoding>,
}

impl ColumnOptions {
    /// Set the general compression.
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = Some(compression);
        self
    }

    /// Set the byte stream split mode.
    pub fn with_byte_stream_split(mut self, mode: ByteStreamSplit) -> Self {
        self.byte_stream_split = Some(mode);
        self
    }

    /// Set the run-length encoding threshold, from 0.0 to 1.0.
    pub fn with_rle_threshold(mut self, threshold: f64) -> Self {
        self.rle_threshold = Some(threshold);
        self
    }

    /// Set the structural encoding.
    pub fn with_structural_encoding(mut self, encoding: StructuralEncoding) -> Self {
        self.structural_encoding = Some(encoding);
        self
    }

    /// The options of `self`, with the options of `defaults` where `self` has none.
    pub fn or(&self, defaults: &ColumnOptions) -> ColumnOptions {
        ColumnOptions {
            compression: self.compression.or(defaults.compression),
            byte_stream_split: self.byte_stream_split.or(defaults.byte_stream_split),
            rle_threshold: self.rle_threshold.or(defaults.rle_threshold),
            structural_encoding: self.structural_encoding.or(defaults.structural_encoding),
        }
    }

    /// The Lance metadata for the leaf fields of a column.
    pub fn leaf_metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        match self.compression {
            Some(Compression::None) => insert(&mut metadata, COMPRESSION_KEY, "none"),
            Some(Compression::Lz4) => insert(&mut metadata, COMPRESSION_KEY, "lz4"),
            Some(Compression::Zstd { level }) => {
                insert(&mut metadata, COMPRESSION_KEY, "zstd");
                if let Some(level) = level {
                    insert(&mut metadata, COMPRESSION_LEVEL_KEY, &level.to_string());
                }
            }
            None => {}
        }
        let mode = self.byte_stream_split.map(|mode| match mode {
            ByteStreamSplit::Off => "off",
            ByteStreamSplit::On => "on",
            ByteStreamSplit::Auto => "auto",
        });
        if let Some(mode) = mode {
            insert(&mut metadata, BYTE_STREAM_SPLIT_KEY, mode);
        }
        if let Some(threshold) = self.rle_threshold {
            insert(&mut metadata, RLE_THRESHOLD_KEY, &threshold.to_string());
        }
        metadata
    }

    /// The Lance metadata for the top-level field of a column.
    pub fn root_metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        let encoding = self.structural_encoding.map(|encoding| match encoding {
            StructuralEncoding::MiniBlock => "miniblock",
            StructuralEncoding::FullZip => "fullzip",
        });
        if let Some(encoding) = encoding {
            insert(&mut metadata, STRUCTURAL_ENCODING_KEY, encoding);
        }
        metadata
    }

    /// The options as table schema metadata, see [`COLUMN_DEFAULTS_PREFIX`].
    pub fn to_table_metadata(&self) -> HashMap<String, String> {
        self.leaf_metadata()
            .into_iter()
            .chain(self.root_metadata())
            .map(|(key, value)| (format!("{COLUMN_DEFAULTS_PREFIX}{key}"), value))
            .collect()
    }

    /// Read the options from table schema metadata. Other keys are ignored.
    pub fn from_table_metadata(metadata: &HashMap<String, String>) -> Result<Self> {
        let get = |key: &str| metadata.get(&format!("{COLUMN_DEFAULTS_PREFIX}{key}"));
        let invalid = |key: &str, value: &str| {
            Error::InvalidTable(format!(
                "column default {key} has the invalid value '{value}'"
            ))
        };
        let level = get(COMPRESSION_LEVEL_KEY)
            .map(|v| v.parse().map_err(|_| invalid(COMPRESSION_LEVEL_KEY, v)))
            .transpose()?;
        let compression = get(COMPRESSION_KEY)
            .map(|v| match v.as_str() {
                "none" => Ok(Compression::None),
                "lz4" => Ok(Compression::Lz4),
                "zstd" => Ok(Compression::Zstd { level }),
                _ => Err(invalid(COMPRESSION_KEY, v)),
            })
            .transpose()?;
        let byte_stream_split = get(BYTE_STREAM_SPLIT_KEY)
            .map(|v| match v.as_str() {
                "off" => Ok(ByteStreamSplit::Off),
                "on" => Ok(ByteStreamSplit::On),
                "auto" => Ok(ByteStreamSplit::Auto),
                _ => Err(invalid(BYTE_STREAM_SPLIT_KEY, v)),
            })
            .transpose()?;
        let rle_threshold = get(RLE_THRESHOLD_KEY)
            .map(|v| v.parse().map_err(|_| invalid(RLE_THRESHOLD_KEY, v)))
            .transpose()?;
        let structural_encoding = get(STRUCTURAL_ENCODING_KEY)
            .map(|v| match v.as_str() {
                "miniblock" => Ok(StructuralEncoding::MiniBlock),
                "fullzip" => Ok(StructuralEncoding::FullZip),
                _ => Err(invalid(STRUCTURAL_ENCODING_KEY, v)),
            })
            .transpose()?;
        Ok(Self {
            compression,
            byte_stream_split,
            rle_threshold,
            structural_encoding,
        })
    }
}

fn insert(metadata: &mut HashMap<String, String>, key: &str, value: &str) {
    metadata.insert(key.to_string(), value.to_string());
}

/// The column defaults of the table. New columns get them.
pub fn column_defaults(dataset: &Dataset) -> Result<ColumnOptions> {
    check_format(dataset)?;
    ColumnOptions::from_table_metadata(&dataset.schema().metadata)
}

/// Replace the column defaults of the table. Existing columns keep their encoding.
pub async fn set_column_defaults(dataset: &mut Dataset, defaults: &ColumnOptions) -> Result<()> {
    check_format(dataset)?;
    let mut values = defaults.to_table_metadata();
    let entries: Vec<(String, Option<String>)> = KEYS
        .iter()
        .map(|key| {
            let key = format!("{COLUMN_DEFAULTS_PREFIX}{key}");
            let value = values.remove(&key);
            (key, value)
        })
        .collect();
    dataset.update_schema_metadata(entries).await?;
    Ok(())
}

/// `schema` with the encoding metadata of each column. The options of a
/// column are its entry in `columns`, then `defaults`.
pub fn with_column_options(
    schema: &Schema,
    defaults: &ColumnOptions,
    columns: &HashMap<String, ColumnOptions>,
) -> Schema {
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|field| {
            let options = match columns.get(field.name()) {
                Some(options) => options.or(defaults),
                None => defaults.clone(),
            };
            let leaf = options.leaf_metadata();
            let mut field = with_leaf_metadata(field, &leaf);
            let mut metadata = field.metadata().clone();
            metadata.extend(options.root_metadata());
            field.set_metadata(metadata);
            field
        })
        .collect();
    Schema::new_with_metadata(fields, schema.metadata().clone())
}

/// `field` with `leaf` added to the metadata of each leaf field.
fn with_leaf_metadata(field: &Field, leaf: &HashMap<String, String>) -> Field {
    let data_type = match field.data_type() {
        DataType::Struct(children) => DataType::Struct(
            children
                .iter()
                .map(|child| with_leaf_metadata(child, leaf))
                .collect(),
        ),
        DataType::List(item) => DataType::List(Arc::new(with_leaf_metadata(item, leaf))),
        _ => {
            let mut metadata = field.metadata().clone();
            metadata.extend(leaf.clone());
            return field.clone().with_metadata(metadata);
        }
    };
    field.clone().with_data_type(data_type)
}

/// `field` without the Lance encoding keys, also in nested fields.
pub fn without_encoding_metadata(field: &Field) -> Field {
    let data_type = match field.data_type() {
        DataType::Struct(children) => DataType::Struct(
            children
                .iter()
                .map(|child| without_encoding_metadata(child))
                .collect(),
        ),
        DataType::List(item) => DataType::List(Arc::new(without_encoding_metadata(item))),
        data_type => data_type.clone(),
    };
    let metadata = field
        .metadata()
        .iter()
        .filter(|(key, _)| !key.starts_with(LANCE_ENCODING_PREFIX))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    field
        .clone()
        .with_data_type(data_type)
        .with_metadata(metadata)
}
