# Python Bindings for lance-nd Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python package `lance_nd` that encodes `xarray.Dataset` objects as nd-arrow PyArrow record batches and writes them to a lance-nd table through PyO3, with one Lance commit per writer at `finish()`.

**Architecture:** A new staged writer `NdWriter` in the lance-nd crate writes Lance data files without a commit and commits all of them in one transaction at `finish()`. A PyO3 crate at `python/` wraps `NdWriter` and a minimal table handle, and receives PyArrow record batches through the Arrow C data interface. A pure-Python encoder in the same package turns xarray datasets into nd-arrow record batches.

**Tech Stack:** Rust 2024 (lance 11, arrow 58, nd-arrow-array), PyO3, maturin, Python with NumPy, PyArrow, xarray, pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-python-bindings-design.md`

## Global Constraints

- The Python package name and import name are `lance_nd`.
- The PyO3 crate lives at `python/` and depends on lance-nd with a path dependency.
- nd-arrow field metadata: `ARROW:extension:name = nd_arrow.array`, `ARROW:extension:metadata = {"version":1}`. Global attributes column: name `__global_attrs__`, `ARROW:extension:name = nd_arrow.global_attributes`, same metadata.
- nd-arrow storage: `Struct { values: List<T> (item field "item", nullable), dim_names: List<Utf8> (items not null), dim_sizes: List<UInt32> (items not null), attrs: Struct {..} (only when a row has attributes) }`.
- Readers see no dataset of a writer before `finish()`. A failed, aborted or dropped writer leaves the table as it was.
- Rust decodes Python batches with `lance_nd::encode::RowDecoder`: a null row means an absent variable. Never `NdRecordBatch::try_from_record_batch`.
- The encoder ignores `Variable.encoding` and `Dataset.encoding`.
- The existing `write()` and `append()` functions stay as they are.
- `uri` is a local path or a URI that Lance opens with no options. No storage options.
- Python calls block and release the GIL while Rust works.
- Commit messages: imperative summary of about 50 characters, no prefixes, no trailers.
- Tests run on Windows and Linux. Use `pathlib` / `tempfile` paths, never hard-coded separators.
- The first commit (Task 1) also adds the spec and this plan.

## Review Focus

- **Fortran-ordered or transposed arrays** (`ds.transpose()`, views): `values` must still be C order for the variable's own `dims`. Test in Task 4.
- **Big-endian dtypes** (`>f4`, `>i2`, common from NetCDF): convert to native byte order, values unchanged. Test in Task 4.
- **Scalars and empty arrays** (rank 0, a dimension of size 0): encode and write without error. Tests in Task 4 and Task 5.
- **A type change for one variable across two `write()` calls in one writer** (`Float32` then `Float64`): an error, and after the error nothing of that writer is committed. Test in Task 2.
- **A writer used after `finish()` or `abort()`**: each call raises, nothing is written. Test in Task 3.

---

### Task 1: Spike: one commit with a schema that grows

This task is a gate. It answers the risk in spec section 7.2. If the spike shows no way, stop and report to the user with the findings. Do not start Task 2.

**Files:**
- Create: `tests/staged_commit.rs`

**Interfaces:**
- Consumes: Lance 11 `InsertBuilder::execute_uncommitted(Vec<RecordBatch>) -> Result<Transaction>`, `CommitBuilder::execute(Transaction) -> Result<Dataset>`, `Operation` variants (`Overwrite`, `Append`, `Merge`).
- Produces: a written finding at the top of `tests/staged_commit.rs` (a short `//!` doc comment) that names the Lance operation and the field-id strategy that Task 2 uses.

- [ ] **Step 1: Write two tests that must pass**

`staged_create_with_new_column`: on an empty temp dir, stage batch A with columns `{a: Int32}` and batch B with `{a: Int32, b: Float64}` without a commit. Assert that `Dataset::open(uri)` fails. Commit once. Assert: `dataset.version().version == 1`, `count_rows() == 2`, schema has `a` and `b`, and `b` is null in the row of batch A.

`staged_append_with_new_column`: create a table with `{a: Int32}` and 1 row (normal write, version 1). Stage batch C `{a: Int32, b: Float64}` without a commit. Assert that the opened dataset is still version 1 with 1 row. Commit once. Assert: version 2, `count_rows() == 2`, schema has `b`, `b` is null in the old row.

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test --test staged_commit`
Expected: FAIL (no commit logic yet).

- [ ] **Step 3: Find the mechanism**

Hypotheses to check, in order:
1. Create: stage each batch with `InsertBuilder` in `WriteMode::Create` against the uri, collect the fragments of all transactions, and commit one `Operation::Overwrite { fragments, schema }` with the merged schema.
2. Append: commit one `Operation::Merge { fragments: existing + staged, schema: merged }`, so old fragments read the new fields as null.
3. Field ids: each staged batch must be written with a Lance schema whose field ids come from the merged schema so far. Two writes must never give one id to two fields. If `InsertBuilder` assigns ids on its own, write through a lower-level fragment writer with an explicit `lance_core::datatypes::Schema`.

- [ ] **Step 4: Run the tests to see them pass**

Run: `cargo test --test staged_commit`
Expected: PASS. Then write the `//!` finding at the top of the file.

- [ ] **Step 5: Commit**

```bash
git add tests/staged_commit.rs docs/superpowers/specs/2026-11-02-python-bindings-design.md docs/superpowers/plans/2026-11-02-python-bindings.md
git commit -m "Prove a staged Lance commit with a growing schema"
```

---

### Task 2: `NdWriter` in lance-nd

**Files:**
- Create: `src/writer.rs`
- Modify: `src/lib.rs` (add `pub mod writer;` and `pub use writer::NdWriter;`)
- Test: `tests/writer.rs`

**Interfaces:**
- Consumes: the mechanism from Task 1; `encode_group`, `table_defaults`, `next_group` patterns in `src/write.rs` (move shared helpers to `pub(crate)` instead of a copy); `WriteOptions` (its `mode` field picks `Create` or `Overwrite` for `NdWriter::create`).
- Produces:
  - `NdWriter::create(uri: &str, options: &WriteOptions) -> Result<NdWriter>` (async). Fails at once when `options.mode` is `Create` and the table exists.
  - `NdWriter::append(uri: &str, options: &WriteOptions) -> Result<NdWriter>` (async). Fails at once when the table does not exist.
  - `NdWriter::write(&mut self, rows: impl Stream<Item = Result<NdRecordBatch>>) -> Result<()>` (async). Writes data files, no commit.
  - `NdWriter::finish(self) -> Result<Dataset>` (async). One commit.
  - `NdWriter::abort(self)`. Commits nothing. A drop does the same.

- [ ] **Step 1: Write the failing tests in `tests/writer.rs`**

Reuse the `sea_surface` style helper from `tests/dataset.rs` (copy it; integration tests do not share modules).

- `create_is_invisible_until_finish`: create, write 2 rows; `Dataset::open` fails; `finish()`; version is 1, `count_rows() == 2`.
- `append_is_one_version`: create a table with `write()` (version N); `NdWriter::append`, write 1 row, write 1 row; open shows version N; `finish()`; version is N + 1, 2 more rows.
- `new_variable_in_second_write`: one writer, first write without variable `extra`, second write with it; after `finish()`, `variables()` holds `extra`, and a `scan()` of the first row has no `extra`.
- `abort_leaves_table`: append writer, write, `abort()`; table version and row count unchanged. Same with a writer that is dropped.
- `create_without_rows_fails`: `create` then `finish()` with no write returns `Error::InvalidArgument`; no table exists. Same with `mode: Overwrite` on an existing table, and the old table is unchanged.
- `append_without_rows_is_no_commit`: `append` then `finish()`; returns the dataset at the same version.
- `type_change_across_writes_fails` (Review Focus): first write `sst` as `Float64`, second as `Float32`; the second `write()` or `finish()` returns an error; the table does not exist (create case) or is unchanged (append case).
- `create_existing_fails`: `NdWriter::create` with `mode: Create` on an existing table fails before any write.
- `overwrite_replaces_in_one_commit`: `mode: Overwrite`, write 1 row, `finish()`; the table has only that row.
- `concurrent_schema_change`: two `NdWriter::append` writers on one table; each writes a row with a different new variable. Finish the first. Finish the second: it either returns `Ok` with both variables in the schema and all rows readable, or returns an error and the table stays at the version of the first commit. Write in a comment which of the two Lance 11 does.

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test --test writer`
Expected: FAIL to compile (`NdWriter` not defined).

- [ ] **Step 3: Implement `NdWriter` in `src/writer.rs`**

Keep the merged table schema in the writer and grow it on each `write()`. Check new fields against it with `merge_schemas` from `src/schema.rs`, so type conflicts fail in `write()`. Store column defaults and options the same way `write()` does. Doc comments in the style of `src/write.rs`.

- [ ] **Step 4: Run all Rust tests**

Run: `cargo test`
Expected: PASS, including `tests/dataset.rs` and `tests/staged_commit.rs`.

- [ ] **Step 5: Commit**

```bash
git add src/writer.rs src/lib.rs src/write.rs tests/writer.rs
git commit -m "Add NdWriter with one commit at finish"
```

---

### Task 3: PyO3 crate with `Writer` and `Table`

**Files:**
- Modify: `Cargo.toml` (add `[workspace] members = [".", "python"]`)
- Modify: `.gitignore` (add `.venv/`, `__pycache__/`, `*.pyd`, `*.so`, `.pytest_cache/`)
- Create: `python/Cargo.toml` (crate `lance-nd-python`, `crate-type = ["cdylib"]`, lib name `_lance_nd`)
- Create: `python/pyproject.toml` (build backend maturin, `module-name = "lance_nd._lance_nd"`, `python-source = "."`, dependencies `numpy`, `pyarrow`, `xarray`; dev extra `pytest`)
- Create: `python/src/lib.rs`
- Create: `python/lance_nd/__init__.py` (re-exports only, for now)
- Create: `python/tests/conftest.py` (fixture `table_uri(tmp_path) -> str`, and `nd_batch()` that builds one valid nd-arrow `pyarrow.RecordBatch` by hand: a `(time=2)` `Float64` variable `sst` with `units = "K"`)
- Test: `python/tests/test_native.py`

**Interfaces:**
- Consumes: `NdWriter` (Task 2), `lance_nd::encode::RowDecoder`, `lance_nd::scan`, `lance_nd::variables`, `lance_nd::encode::encode_rows`.
- Produces, in module `lance_nd._lance_nd`:
  - `class LanceNdError(Exception)`.
  - `class Writer` with `@staticmethod create(uri: str, mode: str = "create", column_defaults: dict | None = None, column_options: dict[str, dict] | None = None) -> Writer`, `@staticmethod append(uri: str, column_options: dict[str, dict] | None = None) -> Writer`, `write_batch(batch: pyarrow.RecordBatch) -> None`, `finish() -> Table`, `abort() -> None`. After `finish()` or `abort()`, each method raises `LanceNdError("writer is closed")`.
  - `class Table` with `@staticmethod open(uri: str) -> Table`, properties `uri: str`, `version: int`, methods `count_rows() -> int`, `variables() -> list[str]`.
  - `_scan_batch(uri: str) -> pyarrow.RecordBatch`: private, for tests only; the whole table as one nd-arrow batch through `scan` and `encode_rows`.
  - Option dicts use the keys of `ColumnOptions`: `compression` (`"zstd"`, `"lz4"`, `"none"`), `compression_level` (int), `byte_stream_split` (bool), `rle` (bool), `structural_encoding` (`"miniblock"`, `"fullzip"`). Match the exact names to `src/options.rs`; an unknown key raises `ValueError`.

- [ ] **Step 1: Set up the Python environment**

Run (from `python/`): `python -m venv .venv`, then `.venv/Scripts/python -m pip install maturin pytest numpy pyarrow xarray` (on Linux: `.venv/bin/python`).
Expected: installs without errors.

- [ ] **Step 2: Write the failing tests in `python/tests/test_native.py`**

- `test_create_write_finish`: `Writer.create(uri)`, `write_batch(nd_batch())`, `Table.open(uri)` raises `LanceNdError`; `finish()` returns a `Table` with `version == 1`, `count_rows() == 1`, `variables() == ["sst"]`.
- `test_round_trip_values`: after the write, `_scan_batch(uri)` column `sst` has `values == [..]` from `nd_batch()`, `dim_names == ["time"]`, `dim_sizes == [2]`, `attrs.units == "K"`.
- `test_append`: create and finish, then `Writer.append(uri)`, write, finish; `version == 2`, `count_rows() == 2`.
- `test_abort`: append writer, write, `abort()`; table unchanged.
- `test_closed_writer_raises` (Review Focus): after `finish()`, `write_batch`, `finish` and `abort` each raise `LanceNdError`; after `abort()`, the same.
- `test_invalid_batch_raises`: a plain `pyarrow.RecordBatch` without nd metadata raises `LanceNdError`.
- `test_unknown_option_raises`: `Writer.create(uri, column_defaults={"bogus": 1})` raises `ValueError`.

- [ ] **Step 3: Run them to see them fail**

Run (from `python/`): `.venv/Scripts/python -m pytest tests/test_native.py -v`
Expected: FAIL with `ModuleNotFoundError: lance_nd._lance_nd`.

- [ ] **Step 4: Implement `python/src/lib.rs`**

One process-wide Tokio runtime (`OnceLock<tokio::runtime::Runtime>`). Each method runs `runtime.block_on` inside `py.allow_threads`. The writer holds `Option<NdWriter>`; `finish`/`abort` take it out. Pick the PyO3 version and the Arrow C data bridge (`arrow` `pyarrow` support or `pyo3-arrow`) so that both use one PyO3 version.

- [ ] **Step 5: Build and run the tests**

Run (from `python/`): `.venv/Scripts/python -m maturin develop`, then `.venv/Scripts/python -m pytest tests/test_native.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock .gitignore python/Cargo.toml python/pyproject.toml python/src/lib.rs python/lance_nd/__init__.py python/tests/conftest.py python/tests/test_native.py
git commit -m "Add PyO3 bindings for NdWriter and tables"
```

---

### Task 4: xarray encoder

**Files:**
- Create: `python/lance_nd/_encode.py`
- Modify: `python/lance_nd/__init__.py` (export `encode`)
- Test: `python/tests/test_encode.py`

**Interfaces:**
- Consumes: nothing from Rust for the encoder itself; the tests use `Writer` and `_scan_batch` (Task 3) to check that Rust accepts the output.
- Produces: `encode(datasets: xarray.Dataset | Sequence[xarray.Dataset]) -> pyarrow.RecordBatch`, one row per dataset, following spec section 6 (structure 6.1, value types 6.2, attribute types 6.3, `encoding` ignored 6.4, one copy 6.5).

- [ ] **Step 1: Write the failing tests in `python/tests/test_encode.py`**

Build small datasets inline. Each test asserts on the PyArrow output, and the tests marked (Rust) also write the batch with `Writer.create`/`finish` and compare `_scan_batch` output.

- `test_structure` (Rust): `ds` with `time (time)`, `lat (lat)`, `sst (time, lat)` Float64, `attrs={"units": "K"}` on `sst`, `ds.attrs={"title": "t"}`. Assert: columns `time`, `lat`, `sst`, `__global_attrs__`; each field has the extension metadata from Global Constraints; `sst` `dim_names == ["time", "lat"]`, `dim_sizes == [2, 3]`; values in C order; `__global_attrs__.title == "t"`.
- `test_no_attrs_field_when_empty`: no variable has attrs; the `sst` storage type has no `attrs` child; no `__global_attrs__` column.
- `test_value_types`: one parametrized test over the rows of spec table 6.2: `bool`, each int and uint width, `float16/32/64`, `datetime64[ns]` → `timestamp("ns")`, `timedelta64[s]` → `duration("s")`, `U`, `S`, `object` strings → `string()`. `complex128` raises `TypeError` with the variable name in the message.
- `test_missing_values`: NaN in a float array, NaT in a datetime array, `None` in an object string array each become a null item; other items unchanged.
- `test_attr_types`: one parametrized test over spec table 6.3: `str`, `bool`, `np.bool_`, `int` → `int64`, `float` → `float64`, `np.int16(3)` → `int16`, `np.array([1.0, 2.0])` → `list<float64>`, `[1, 2]` → `list<int64>`. A `dict` value raises `TypeError` with the key name.
- `test_multiple_datasets` (Rust): `encode([ds_a, ds_b])` where only `ds_a` has `extra`; 2 rows; `extra` is a null row in row 1; after the write, `_scan_batch` row 1 has no `extra` value (null).
- `test_type_conflict_between_datasets`: `sst` Float64 in `ds_a` and Float32 in `ds_b` raises `TypeError`; an attribute key with `str` in one and `int` in the other raises `TypeError`.
- `test_cf_coordinates`: `ds` with `temp (y, x)`, non-index coordinates `lat2d (y, x)`, `lon2d (y, x)`, and a scalar coordinate `ref_time ()`. `temp` attrs get `coordinates == "lat2d lon2d ref_time"` (a scalar fits in every variable); index coordinates are never listed. A coordinate that no data variable can list goes to `__global_attrs__.coordinates`. Existing `coordinates` key in `attrs` raises `ValueError`.
- `test_encoding_ignored`: `ds.sst.encoding = {"_FillValue": -999, "scale_factor": 0.1}`; output attrs have no such keys.
- `test_reserved_name`: a variable named `__global_attrs__` raises `ValueError`.
- `test_fortran_order` (Review Focus, Rust): `sst` built from `np.asfortranarray(...)`, and a second case from `ds.transpose("lat", "time")`; values match `var.values.ravel(order="C")` for the variable's own dims.
- `test_big_endian` (Review Focus, Rust): `sst` with dtype `>f4`; Arrow type is `float32` and values are equal.
- `test_scalar_and_empty` (Review Focus, Rust): a rank-0 variable (`dim_names == []`, one value) and a variable with a dimension of size 0 (zero values) both encode and write.
- `test_dask_is_loaded`: skip when `dask` is not installed; a chunked variable encodes the same as the loaded one.

- [ ] **Step 2: Run them to see them fail**

Run (from `python/`): `.venv/Scripts/python -m pytest tests/test_encode.py -v`
Expected: FAIL with `ImportError: cannot import name 'encode'`.

- [ ] **Step 3: Implement `encode` in `python/lance_nd/_encode.py`**

Build each column as a `pyarrow.StructArray` with an explicit validity mask for absent variables, and the field metadata on `pyarrow.field(...)` (no registered extension type is needed). Use `np.ascontiguousarray` on the variable data, then `astype(dtype.newbyteorder("="))` for non-native byte order, then `pyarrow.array(flat, from_pandas=True)` for NaN and NaT to null.

- [ ] **Step 4: Run the tests**

Run (from `python/`): `.venv/Scripts/python -m pytest tests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lance_nd/_encode.py python/lance_nd/__init__.py python/tests/test_encode.py
git commit -m "Add xarray to nd-arrow encoder"
```

---

### Task 5: Public Python API

**Files:**
- Create: `python/lance_nd/_api.py`
- Modify: `python/lance_nd/__init__.py` (export `create`, `append`, `open`, `encode`, `Table`, `Writer`, `LanceNdError`)
- Test: `python/tests/test_api.py`

**Interfaces:**
- Consumes: `_lance_nd.Writer`, `_lance_nd.Table`, `_lance_nd.LanceNdError` (Task 3), `encode` (Task 4).
- Produces:
  - `create(uri: str, mode: str = "create", column_defaults: dict | None = None, column_options: dict | None = None) -> Writer`
  - `append(uri: str, column_options: dict | None = None) -> Writer`
  - `open(uri: str) -> Table` (the `_lance_nd.Table` class)
  - `class Writer` (Python wrapper): `write(data: xarray.Dataset | Sequence[xarray.Dataset] | pyarrow.RecordBatch) -> None`, `finish() -> Table`, `abort() -> None`, `__enter__` returns itself, `__exit__` calls `finish()` on success and `abort()` on an exception (and does not swallow the exception).

- [ ] **Step 1: Write the failing tests in `python/tests/test_api.py`**

- `test_spec_example`: the code block of spec section 5 runs as written (with temp paths): `create`, `write(ds_a)`, `write([ds_b, ds_c])`, `write(encode(ds_d))`, `finish()` gives `count_rows() == 4`, `version == 1`.
- `test_append_api`: `append(uri)` on a finished table adds rows in one new version; `append` on a missing table raises `LanceNdError`.
- `test_context_manager_finishes`: a `with create(uri) as w: w.write(ds)` block; afterwards `open(uri).count_rows() == 1`.
- `test_context_manager_aborts`: an exception inside a `with append(uri)` block propagates; the table version is unchanged.
- `test_empty_with_block`: `with create(uri): pass` raises `LanceNdError` and no table exists; `with append(uri): pass` leaves the version unchanged.
- `test_overwrite`: `create(uri, mode="overwrite")` on an existing table, write 1 dataset, finish; `count_rows() == 1`.
- `test_create_existing_raises`: `create(uri)` on an existing table raises `LanceNdError`.
- `test_write_rejects_other_types`: `write(42)` raises `TypeError`.

- [ ] **Step 2: Run them to see them fail**

Run (from `python/`): `.venv/Scripts/python -m pytest tests/test_api.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `python/lance_nd/_api.py`**

`write` calls `encode` for xarray input and passes record batches to `_lance_nd.Writer.write_batch` unchanged.

- [ ] **Step 4: Run all tests**

Run: `cargo test`, then (from `python/`) `.venv/Scripts/python -m maturin develop` and `.venv/Scripts/python -m pytest tests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lance_nd/_api.py python/lance_nd/__init__.py python/tests/test_api.py
git commit -m "Add the public Python API"
```

---

### Task 6: Documentation

**Files:**
- Create: `python/README.md` (install with maturin, the spec section 5 example, the encoder rules of spec section 6 as short tables)
- Modify: `README.md` (one short "Python" section that links to `python/README.md`; add `NdWriter` to the table of main functions)

**Interfaces:**
- Consumes: the final API of Tasks 2 and 5.
- Produces: nothing for other tasks.

- [ ] **Step 1: Write the docs**

Use ASD-STE100 style, as the existing README: active voice, short sentences, no em dashes.

- [ ] **Step 2: Check the example**

Run the `python/README.md` example with the venv Python against a temp path.
Expected: it runs without errors.

- [ ] **Step 3: Commit**

```bash
git add python/README.md README.md
git commit -m "Document the Python package"
```
