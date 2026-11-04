//! Delete rows, compact the table and clean up old versions.
//!
//! A delete marks rows in deletion files and keeps the data files. A
//! compaction rewrites small fragments and fragments with many deleted rows.
//! Both make a new table version. The files of old versions stay on disk
//! until [`cleanup_old_versions`] removes them.

use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{RecordBatch, RecordBatchIterator};
use arrow::compute::concat_batches;
use arrow::datatypes::{Schema, SchemaRef};
use arrow::error::ArrowError;
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::cleanup::RemovalStats;
use lance::dataset::fragment::write::FragmentCreateBuilder;
use lance::dataset::optimize::CompactionMetrics;
use lance::dataset::transaction::{Operation, RewriteGroup, Transaction};
use lance::dataset::write::delete::DeleteBuilder;
use lance::dataset::{CommitBuilder, WriteMode, WriteParams};
use lance::deps::datafusion::logical_expr::Expr;
use lance::table::format::Fragment;
use lance_core::datatypes::Schema as LanceSchema;

use crate::error::Result;
use crate::schema::{conform_batch, fill_hidden_nulls};
use crate::write::FILE_VERSION;

/// Delete the rows that match `predicate`, a Lance SQL filter.
///
/// Lance gets `predicate` with no change, so it must be exact for a row,
/// for example `__global_attrs__.id = 'a'` or `sst.attrs.units = 'K'`.
/// Returns the number of deleted rows.
pub async fn delete(dataset: &mut Dataset, predicate: &str) -> Result<u64> {
    let builder = DeleteBuilder::new(Arc::new(dataset.clone()), predicate);
    run_delete(dataset, builder).await
}

/// Delete the rows that match `expr`, a DataFusion filter on the Lance
/// table columns.
///
/// Lance gets `expr` with no change. Unlike [`crate::NdScanner::filter_expr`],
/// a delete does not change value parts into stats filters, because a stats
/// filter keeps rows that do not match. Returns the number of deleted rows.
pub async fn delete_expr(dataset: &mut Dataset, expr: Expr) -> Result<u64> {
    let builder = DeleteBuilder::from_expr(Arc::new(dataset.clone()), expr);
    run_delete(dataset, builder).await
}

async fn run_delete(dataset: &mut Dataset, builder: DeleteBuilder) -> Result<u64> {
    let result = builder.execute().await?;
    *dataset = Arc::unwrap_or_clone(result.new_dataset);
    Ok(result.num_deleted_rows)
}

/// Options for [`compact`].
#[derive(Debug, Clone)]
pub struct CompactOptions {
    /// The size of a fragment on disk that a compaction aims for. A fragment
    /// below half this size is merged with its small neighbors. A fragment
    /// above twice this size is split. The default is 128 MiB.
    pub target_fragment_bytes: u64,
    /// A fragment with at least this fraction of deleted rows is rewritten
    /// without them. The default is 0.1. A value above 1 never rewrites a
    /// fragment for its deleted rows.
    pub deleted_fraction: f32,
}

impl Default for CompactOptions {
    fn default() -> Self {
        Self {
            target_fragment_bytes: 128 << 20,
            deleted_fraction: 0.1,
        }
    }
}

/// Merge small fragments, split large fragments and remove deleted rows.
/// Keeps the row order.
///
/// The plan uses the size of each fragment on disk. Each run of
/// neighboring fragments that need work becomes fragments of about the
/// same size, near the target. A row is never split, so a row larger than
/// the target is a fragment on its own. lance-nd holds about one new
/// fragment in memory at a time. Without work to do, the table keeps its
/// version.
pub async fn compact(dataset: &mut Dataset, options: &CompactOptions) -> Result<CompactionMetrics> {
    let fragments: Vec<Fragment> = dataset.fragments().as_ref().clone();
    let mut pieces = Vec::with_capacity(fragments.len());
    for fragment in &fragments {
        pieces.push(piece_of(dataset, fragment).await?);
    }
    let target = options.target_fragment_bytes.max(1);
    let tasks = plan(&pieces, target, options.deleted_fraction);
    let mut metrics = CompactionMetrics::default();
    if tasks.is_empty() {
        return Ok(metrics);
    }
    let read_version = dataset.version().version;
    let mut groups = Vec::with_capacity(tasks.len());
    for task in tasks {
        let old = &fragments[task.clone()];
        let new_fragments = rewrite(dataset, old, &pieces[task], target).await?;
        metrics.fragments_removed += old.len();
        metrics.files_removed += old
            .iter()
            .map(|f| f.files.len() + usize::from(f.deletion_file.is_some()))
            .sum::<usize>();
        metrics.fragments_added += new_fragments.len();
        metrics.files_added += new_fragments.len();
        groups.push(RewriteGroup {
            old_fragments: old.to_vec(),
            new_fragments,
        });
    }
    // New fragments need ids that no other writer takes.
    let count: usize = groups.iter().map(|g| g.new_fragments.len()).sum();
    let reserve = Operation::ReserveFragments {
        num_fragments: count as u32,
    };
    let reserved = CommitBuilder::new(Arc::new(dataset.clone()))
        .execute(Transaction::new(read_version, reserve, None))
        .await?;
    let end = u64::from(reserved.manifest().max_fragment_id.unwrap_or(0)) + 1;
    let ids = end - count as u64..end;
    for (fragment, id) in groups
        .iter_mut()
        .flat_map(|g| g.new_fragments.iter_mut())
        .zip(ids)
    {
        fragment.id = id;
    }
    let rewrite = Operation::Rewrite {
        groups,
        rewritten_indices: Vec::new(),
        frag_reuse_index: None,
    };
    *dataset = CommitBuilder::new(Arc::new(reserved))
        .execute(Transaction::new(read_version, rewrite, None))
        .await?;
    Ok(metrics)
}

/// The size of a fragment on disk and its row counts.
#[derive(Debug, Clone, Copy)]
struct Piece {
    bytes: u64,
    rows: u64,
    deleted: u64,
}

impl Piece {
    /// The estimated bytes of one row: the rows of a fragment count the same.
    fn row_bytes(&self) -> f64 {
        self.bytes as f64 / self.rows.max(1) as f64
    }

    fn live_rows(&self) -> u64 {
        self.rows.saturating_sub(self.deleted)
    }
}

async fn piece_of(dataset: &Dataset, fragment: &Fragment) -> Result<Piece> {
    let mut bytes = 0;
    for file in &fragment.files {
        bytes += match file.file_size_bytes.get() {
            Some(size) => size.get(),
            None => {
                let path = dataset.data_dir().join(file.path.as_str());
                dataset.object_store(file.base_id).await?.size(&path).await?
            }
        };
    }
    let deleted = fragment
        .deletion_file
        .as_ref()
        .and_then(|file| file.num_deleted_rows)
        .unwrap_or(0);
    Ok(Piece {
        bytes,
        rows: fragment.physical_rows.unwrap_or(0) as u64,
        deleted: deleted as u64,
    })
}

/// The runs of neighboring fragments to rewrite, as index ranges.
///
/// A fragment needs work when it is below half of `target`, above twice
/// `target` with more than one row, or has at least `deleted_fraction` of
/// its rows deleted. A run of one small fragment alone has nothing to
/// merge with, so it stays.
fn plan(pieces: &[Piece], target: u64, deleted_fraction: f32) -> Vec<Range<usize>> {
    let small = |p: &Piece| p.bytes < target / 2;
    let rewrite = |p: &Piece| {
        let large = p.bytes > target.saturating_mul(2) && p.rows > 1;
        let deleted = deleted_fraction <= 1.0
            && p.deleted > 0
            && p.deleted as f32 >= deleted_fraction * p.rows as f32;
        large || deleted
    };
    let marked = |p: &Piece| small(p) || rewrite(p);
    let mut tasks = Vec::new();
    let mut start = 0;
    while start < pieces.len() {
        if !marked(&pieces[start]) {
            start += 1;
            continue;
        }
        let mut end = start;
        while end < pieces.len() && marked(&pieces[end]) {
            end += 1;
        }
        if end - start > 1 || rewrite(&pieces[start]) {
            tasks.push(start..end);
        }
        start = end;
    }
    tasks
}

/// The number of new fragments for `bytes` in `rows` rows: the nearest to
/// `bytes / target`, at least one, at most one per row.
fn outputs(bytes: u64, rows: u64, target: u64) -> u64 {
    let count = (bytes as f64 / target as f64).round() as u64;
    count.clamp(1, rows.max(1))
}

/// The live rows of `fragments`, written again as new fragments without
/// ids, each of about the same size on disk.
///
/// The new fragments hold only the top-level columns that `fragments` hold,
/// as a writer writes them. The others read as null.
async fn rewrite(
    dataset: &Dataset,
    fragments: &[Fragment],
    pieces: &[Piece],
    target: u64,
) -> Result<Vec<Fragment>> {
    let held: HashSet<i32> = fragments
        .iter()
        .flat_map(|fragment| fragment.files.iter())
        .flat_map(|file| file.fields.iter().copied())
        .collect();
    let names: Vec<&str> = dataset
        .schema()
        .fields
        .iter()
        .filter(|field| holds(&held, field))
        .map(|field| field.name.as_str())
        .collect();
    let lance_schema = dataset.schema().project(&names)?;
    let writer = PartWriter {
        dataset,
        arrow_schema: Arc::new(Schema::from(&lance_schema)),
        lance_schema,
    };

    let bytes: f64 = pieces
        .iter()
        .map(|p| p.row_bytes() * p.live_rows() as f64)
        .sum();
    let rows: u64 = pieces.iter().map(Piece::live_rows).sum();
    let count = outputs(bytes as u64, rows, target);
    // Cut at each multiple of the mean part size, so the parts come out even.
    let part_bytes = bytes / count as f64;
    let mut cuts = 1;
    let mut written = 0.0;
    let mut pending: Vec<RecordBatch> = Vec::new();
    let mut new_fragments = Vec::new();
    for (fragment, piece) in fragments.iter().zip(pieces) {
        let row_bytes = piece.row_bytes().max(f64::MIN_POSITIVE);
        let mut scanner = dataset.scan();
        scanner.project(&names)?;
        scanner.with_fragments(vec![fragment.clone()]);
        // Small batches, so that a cut can fall between them.
        scanner.batch_size(64);
        let mut stream = scanner.try_into_stream().await?;
        while let Some(batch) = stream.try_next().await? {
            let mut batch = writer.conform(&batch)?;
            while batch.num_rows() > 0 {
                let take = if cuts < count {
                    let left = part_bytes * cuts as f64 - written;
                    ((left / row_bytes).ceil() as usize).clamp(1, batch.num_rows())
                } else {
                    batch.num_rows()
                };
                pending.push(batch.slice(0, take));
                written += row_bytes * take as f64;
                batch = batch.slice(take, batch.num_rows() - take);
                if cuts < count && written >= part_bytes * cuts as f64 {
                    new_fragments.extend(writer.write(&std::mem::take(&mut pending)).await?);
                    cuts += 1;
                }
            }
        }
    }
    new_fragments.extend(writer.write(&pending).await?);
    Ok(new_fragments)
}

/// Writes rows as new fragments of one schema.
struct PartWriter<'a> {
    dataset: &'a Dataset,
    lance_schema: LanceSchema,
    arrow_schema: SchemaRef,
}

impl PartWriter<'_> {
    /// `batch` with the columns and the null rules of the new fragments.
    fn conform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let batch = conform_batch(batch, &self.arrow_schema)?;
        let columns = self
            .arrow_schema
            .fields()
            .iter()
            .zip(batch.columns())
            .map(|(field, column)| fill_hidden_nulls(column, field))
            .collect::<Result<Vec<_>>>()?;
        Ok(RecordBatch::try_new(self.arrow_schema.clone(), columns)?)
    }

    /// `batches` as fragments: one fragment, or more when the rows do not
    /// fit in one Arrow array.
    async fn write(&self, batches: &[RecordBatch]) -> Result<Vec<Fragment>> {
        let params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(FILE_VERSION),
            ..Default::default()
        };
        let concat = |part: &[RecordBatch]| concat_batches(&self.arrow_schema, part);
        let mut new_fragments = Vec::new();
        // One batch per fragment: Lance 11 fails to read back a fragment of
        // several batches when a struct column has nulls and large lists.
        for batch in split_concat(batches, &concat)? {
            if batch.num_rows() == 0 {
                continue;
            }
            let reader = RecordBatchIterator::new([Ok(batch)], self.arrow_schema.clone());
            new_fragments.extend(
                FragmentCreateBuilder::new(self.dataset.uri())
                    .schema(&self.lance_schema)
                    .write_params(&params)
                    .write_fragments(reader)
                    .await?,
            );
        }
        Ok(new_fragments)
    }
}

/// `batches` joined by `concat` into as few batches as possible, in order.
///
/// A list or a string array holds at most `i32::MAX` values. When the join
/// overflows that limit, each half of `batches` is joined on its own.
fn split_concat(
    batches: &[RecordBatch],
    concat: &impl Fn(&[RecordBatch]) -> Result<RecordBatch, ArrowError>,
) -> Result<Vec<RecordBatch>> {
    match concat(batches) {
        Ok(batch) => Ok(vec![batch]),
        Err(ArrowError::OffsetOverflowError(_)) if batches.len() > 1 => {
            let (first, second) = batches.split_at(batches.len() / 2);
            let mut parts = split_concat(first, concat)?;
            parts.extend(split_concat(second, concat)?);
            Ok(parts)
        }
        Err(error) => Err(error.into()),
    }
}

/// `true` when `held` has the id of `field` or of a field in it. A data file
/// lists the ids of leaf fields, not always of their parents.
fn holds(held: &HashSet<i32>, field: &lance_core::datatypes::Field) -> bool {
    held.contains(&field.id) || field.children.iter().any(|child| holds(held, child))
}

/// Remove the files of the table versions older than `older_than`. The
/// current version always stays.
///
/// A reader that still uses a removed version fails, so keep a margin for
/// long scans.
pub async fn cleanup_old_versions(dataset: &Dataset, older_than: Duration) -> Result<RemovalStats> {
    let older_than = chrono::Duration::from_std(older_than).unwrap_or(chrono::Duration::MAX);
    Ok(dataset.cleanup_old_versions(older_than, None, None).await?)
}

#[cfg(test)]
mod tests {
    use arrow::array::{AsArray, Int64Array};
    use arrow::datatypes::{DataType, Field, Int64Type};

    use super::*;

    fn batch(id: i64) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![id]))]).unwrap()
    }

    fn ids(batch: &RecordBatch) -> Vec<i64> {
        batch.column(0).as_primitive::<Int64Type>().values().to_vec()
    }

    fn piece(bytes: u64, rows: u64, deleted: u64) -> Piece {
        Piece {
            bytes,
            rows,
            deleted,
        }
    }

    #[test]
    fn plan_merges_small_neighbors_and_keeps_others() {
        let pieces = [
            piece(10, 1, 0),
            piece(20, 2, 0),
            piece(100, 5, 0),
            piece(30, 3, 0),
            piece(100, 5, 0),
            piece(10, 1, 0),
            piece(10, 1, 0),
        ];
        assert_eq!(plan(&pieces, 100, 0.1), [0..2, 5..7]);
    }

    #[test]
    fn plan_skips_a_lone_small_fragment() {
        let pieces = [piece(100, 5, 0), piece(10, 1, 0), piece(100, 5, 0)];
        assert!(plan(&pieces, 100, 0.1).is_empty());
    }

    #[test]
    fn plan_splits_a_large_fragment_of_several_rows() {
        let pieces = [piece(100, 5, 0), piece(500, 5, 0), piece(500, 1, 0)];
        let tasks = plan(&pieces, 100, 0.1);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0], 1..2);
    }

    #[test]
    fn plan_rewrites_deleted_rows() {
        let pieces = [piece(100, 10, 1), piece(100, 10, 0), piece(100, 10, 5)];
        assert_eq!(plan(&pieces, 100, 0.1), [0..1, 2..3]);
        assert!(plan(&pieces, 100, 1.5).is_empty());
    }

    #[test]
    fn outputs_follow_the_target() {
        assert_eq!(outputs(1000, 100, 100), 10);
        assert_eq!(outputs(1040, 100, 100), 10);
        assert_eq!(outputs(10, 100, 100), 1);
        assert_eq!(outputs(1000, 3, 100), 3);
        assert_eq!(outputs(0, 0, 100), 1);
    }

    #[test]
    fn concat_splits_on_offset_overflow() {
        let batches: Vec<RecordBatch> = (0..5).map(batch).collect();
        let schema = batches[0].schema();
        // More than two batches overflow, as a large list column does.
        let concat = |part: &[RecordBatch]| {
            if part.len() > 2 {
                return Err(ArrowError::OffsetOverflowError(part.len()));
            }
            concat_batches(&schema, part)
        };
        let parts = split_concat(&batches, &concat).unwrap();
        let parts: Vec<Vec<i64>> = parts.iter().map(ids).collect();
        assert_eq!(parts, [vec![0, 1], vec![2], vec![3, 4]]);
    }

    #[test]
    fn concat_keeps_other_errors() {
        let batches: Vec<RecordBatch> = (0..2).map(batch).collect();
        let concat = |_: &[RecordBatch]| Err(ArrowError::ComputeError("no".into()));
        assert!(split_concat(&batches, &concat).is_err());
    }
}
