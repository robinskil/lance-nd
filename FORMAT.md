# lance-nd table format

Version 1

This document specifies how a Lance table stores nd record batches.
The storage is the `nd_arrow.array` encoding of
[nd-arrow-array](https://github.com/robinskil/nd-arrow-array/blob/main/FORMAT.md).

## 1. Overview

A table is a Lance table. Each row of the table is one nd record batch:
a set of named nd arrow arrays on one grid, plus optional global attributes.
A row can hold a complete nd record batch or a chunk of a larger one.
The format does not separate the two.

Rows can have different arrays and different grids.
Version 1 has no indexes.

## 2. Columns

The table schema is an nd-arrow encoded schema (nd-arrow format, section 7):

| Column | Type | Nullable |
|---|---|---|
| one column per array name | `nd_arrow.array` | yes |
| `__global_attrs__` | `nd_arrow.global_attributes` | yes |

Rules:

1. The array columns hold the union of the array names of all rows.
   One array name has one value type in a table.
2. A null in an array column means that the row does not have that array.
   A reader leaves the array out of the row. It does not decode it as a
   rank-0 null scalar.
3. The children of a null array row are valid: empty `values`, `dim_names`
   and `dim_sizes` lists. Lance rejects nulls in non-nullable fields.
4. The `__global_attrs__` column holds the global attributes of each row.
   A table without global attributes does not have this column.
5. The nd-arrow format specifies `__global_attrs__` as not nullable. Lance can
   only add nullable columns to an existing table, so a writer makes it
   nullable. A null row has no global attributes.
6. The column order has no meaning.
7. The table schema metadata holds the column defaults, see section 6.
   Other keys have no meaning. The field metadata of an array column comes
   from the first row with that array.

## 3. Grid

The format does not store the grid of a row. A reader infers it from the
arrays of the row, as in the nd-arrow format.

## 4. Schema evolution

A write can add array names, attribute keys and global attributes.
The writer adds the new fields as all-null fields first, then appends the rows.
Older rows read the new fields as null, so as absent.

Nested fields need Lance file version 2.2 or later.
A writer must create the table with file version 2.2 or later.

## 5. Identity of rows

The format has no reserved columns for names or chunk positions.
Put them in the global attributes, for example `id` or `chunk_offset`.
A filter on `__global_attrs__.id` then selects the rows with that id.

## 6. Column encoding

Lance reads the encoding of a column from field metadata with the prefix
`lance-encoding:`. The table schema stores this metadata, so each later
append uses the encoding of the column.

| Key | Field | Values |
|---|---|---|
| `lance-encoding:structural-encoding` | top-level column | `miniblock`, `fullzip` |
| `lance-encoding:compression` | each leaf field | `none`, `lz4`, `zstd` |
| `lance-encoding:compression-level` | each leaf field | integer, for `zstd` |
| `lance-encoding:bss` | each leaf field | `off`, `on`, `auto` |
| `lance-encoding:rle-threshold` | each leaf field | 0.0 to 1.0 |

A leaf field is a field that is not a struct or a list, for example the
`item` field of `values`. Each leaf of a column gets the same keys.

The column defaults are table schema metadata keys. Each key is the prefix
`lance_nd.column_defaults.` plus one key of the table above, for example
`lance_nd.column_defaults.lance-encoding:compression = zstd`. A writer
gives the defaults to each column that it adds. Options for one column
override the defaults. A column keeps its encoding when the defaults change.

A reader removes the `lance-encoding:` keys from the arrays and attributes
that it returns.
