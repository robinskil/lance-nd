import pyarrow as pa
import pytest

from lance_nd._lance_nd import LanceNdError, Table, Writer, _scan_batch


def test_create_write_finish(table_uri, nd_batch):
    writer = Writer.create(table_uri)
    writer.write_batch(nd_batch)
    with pytest.raises(LanceNdError):
        Table.open(table_uri)
    table = writer.finish()
    assert table.version == 1
    assert table.count_rows() == 1
    assert table.variables() == ["sst"]
    assert table.uri == table_uri


def test_round_trip_values(table_uri, nd_batch):
    writer = Writer.create(table_uri)
    writer.write_batch(nd_batch)
    writer.finish()
    sst = _scan_batch(table_uri).column("sst")[0].as_py()
    assert sst["values"] == [1.0, 2.0]
    assert sst["dim_names"] == ["time"]
    assert sst["dim_sizes"] == [2]
    assert sst["attrs"]["units"] == "K"


def test_append(table_uri, nd_batch):
    writer = Writer.create(table_uri)
    writer.write_batch(nd_batch)
    writer.finish()
    writer = Writer.append(table_uri)
    writer.write_batch(nd_batch)
    table = writer.finish()
    assert table.version == 2
    assert table.count_rows() == 2


def test_abort(table_uri, nd_batch):
    writer = Writer.create(table_uri)
    writer.write_batch(nd_batch)
    writer.finish()
    writer = Writer.append(table_uri)
    writer.write_batch(nd_batch)
    writer.abort()
    table = Table.open(table_uri)
    assert table.version == 1
    assert table.count_rows() == 1


@pytest.mark.parametrize("close", ["finish", "abort"])
def test_closed_writer_raises(table_uri, nd_batch, close):
    writer = Writer.create(table_uri)
    writer.write_batch(nd_batch)
    getattr(writer, close)()
    with pytest.raises(LanceNdError, match="writer is closed"):
        writer.write_batch(nd_batch)
    with pytest.raises(LanceNdError, match="writer is closed"):
        writer.finish()
    with pytest.raises(LanceNdError, match="writer is closed"):
        writer.abort()


def test_invalid_batch_raises(table_uri):
    plain = pa.RecordBatch.from_pydict({"x": [1, 2]})
    writer = Writer.create(table_uri)
    with pytest.raises(LanceNdError):
        writer.write_batch(plain)


def test_unknown_option_raises(table_uri):
    with pytest.raises(ValueError, match="bogus"):
        Writer.create(table_uri, column_defaults={"bogus": 1})


def test_options_are_accepted(table_uri, nd_batch):
    writer = Writer.create(
        table_uri,
        column_defaults={"compression": "zstd", "compression_level": 3},
        column_options={"sst": {"byte_stream_split": "on", "structural_encoding": "miniblock"}},
    )
    writer.write_batch(nd_batch)
    assert writer.finish().count_rows() == 1
