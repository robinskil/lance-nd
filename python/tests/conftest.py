import pyarrow as pa
import pytest

ND_ARRAY = {
    b"ARROW:extension:name": b"nd_arrow.array",
    b"ARROW:extension:metadata": b'{"version":1}',
}
GLOBAL_ATTRIBUTES = {
    b"ARROW:extension:name": b"nd_arrow.global_attributes",
    b"ARROW:extension:metadata": b'{"version":1}',
}


@pytest.fixture
def table_uri(tmp_path):
    return str(tmp_path / "table.lance")


@pytest.fixture
def nd_batch():
    """One dataset: a (time=2) Float64 variable sst with units K, title t."""
    values = pa.array([[1.0, 2.0]], type=pa.list_(pa.field("item", pa.float64(), True)))
    dim_names = pa.array([["time"]], type=pa.list_(pa.field("item", pa.string(), False)))
    dim_sizes = pa.array([[2]], type=pa.list_(pa.field("item", pa.uint32(), False)))
    attrs = pa.StructArray.from_arrays([pa.array(["K"])], names=["units"])
    sst = pa.StructArray.from_arrays(
        [values, dim_names, dim_sizes, attrs],
        fields=[
            pa.field("values", values.type, False),
            pa.field("dim_names", dim_names.type, False),
            pa.field("dim_sizes", dim_sizes.type, False),
            pa.field("attrs", attrs.type, False),
        ],
    )
    global_attrs = pa.StructArray.from_arrays([pa.array(["t"])], names=["title"])
    schema = pa.schema(
        [
            pa.field("sst", sst.type, True, metadata=ND_ARRAY),
            pa.field("__global_attrs__", global_attrs.type, True, metadata=GLOBAL_ATTRIBUTES),
        ]
    )
    return pa.RecordBatch.from_arrays([sst, global_attrs], schema=schema)
