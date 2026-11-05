# Fuzz tests

`row_decoder` builds record batches with nd columns of any content and
gives them to `RowDecoder`. A bad batch must return an error, never panic.

```bash
cargo +nightly fuzz run row_decoder -- -max_total_time=300
```
