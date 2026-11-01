# lance-nd table format

Version 1

This document specifies how a Lance dataset stores N-dimensional datasets.
The storage is the `nd_arrow.array` encoding of
[nd-arrow-array](https://github.com/robinskil/nd-arrow-array/blob/main/FORMAT.md).

## 1. Overview

A table is a Lance dataset. Each row of the table is one nd record batch:
a set of variables on one grid, like an xarray `Dataset`. A row can hold a
whole dataset or a chunk of a dataset. The format does not separate the two.

Rows can have different variables and different grids.
Version 1 has no indexes.

## 2. Columns

The table schema is an nd-arrow encoded schema (nd-arrow format, section 7):

| Column | Type | Nullable |
|---|---|---|
| one column per variable | `nd_arrow.array` | yes |
| `__global_attrs__` | `nd_arrow.global_attributes` | yes |

Rules:

1. The variable columns hold the union of the variables of all rows.
   One variable has one value type in a table.
2. A null variable in a row means that the row does not have that variable.
   A reader leaves the variable out of the row. It does not decode it as a
   rank-0 null scalar.
3. The children of a null variable row are valid: empty `values`, `dim_names`
   and `dim_sizes` lists. Lance rejects nulls in non-nullable fields.
4. The `__global_attrs__` column holds the global attributes of each row.
   A table without global attributes does not have this column.
5. The nd-arrow format specifies `__global_attrs__` as not nullable. Lance can
   only add nullable columns to an existing table, so a writer makes it
   nullable. A null row has no global attributes.
6. The column order has no meaning.
7. The table schema metadata holds the column defaults, see section 6.
   Other keys have no meaning. The field metadata of a variable comes from
   the first row with that variable.

## 3. Grid

The format does not store the grid of a row. A reader infers it from the
variables of the row, as in the nd-arrow format.

## 4. Schema evolution

A write can add variables, attribute keys and global attributes.
The writer adds the new fields as all-null fields first, then appends the rows.
Older rows read the new fields as null, so as absent.

Nested fields need Lance file version 2.2 or later.
A writer must create the table with file version 2.2 or later.

## 5. Identity of rows

The format has no reserved columns for dataset names or chunk positions.
Put them in the global attributes, for example `id` or `chunk_offset`.
A filter on `__global_attrs__.id` then selects the rows of one dataset.

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

A reader removes the `lance-encoding:` keys from the variables and
attributes that it returns.
