//! Delete rows, compact the table and clean up old versions.
//!
//! A delete marks rows in deletion files and keeps the data files. A
//! compaction rewrites small fragments and fragments with many deleted rows.
//! Both make a new table version. The files of old versions stay on disk
//! until [`cleanup_old_versions`] removes them.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{RecordBatch, RecordBatchIterator};
use arrow::compute::concat_batches;
use arrow::datatypes::{Schema, SchemaRef};
use arrow::error::ArrowError;
use futures::{StreamExt, TryStreamExt};
use lance::Dataset;
use lance::dataset::fragment::write::FragmentCreateBuilder;
use lance::dataset::fragment::{FileFragment, FragReadConfig};
use lance::dataset::transaction::{Operation, RewriteGroup, Transaction};
use lance::dataset::write::delete::DeleteBuilder;
use lance::dataset::{CommitBuilder, WriteMode, WriteParams};
use lance::deps::datafusion::logical_expr::Expr;
use lance::table::format::{Fragment, Manifest};
use lance_core::datatypes::Schema as LanceSchema;

use crate::error::Result;
use crate::format::check_format;
use crate::schema::{conform_batch, fill_hidden_nulls};
use crate::write::FILE_VERSION;

/// Delete the rows that match `predicate`, a Lance SQL filter.
///
/// Lance gets `predicate` with no change, so it must be exact for a row,
/// for example `__global_attrs__.id = 'a'` or `sst.attrs.units = 'K'`.
/// Returns the number of deleted rows.
#[tracing::instrument(skip_all, fields(uri = %dataset.uri(), predicate = %predicate), err)]
pub async fn delete(dataset: &mut Dataset, predicate: &str) -> Result<u64> {
    check_format(dataset)?;
    let builder = DeleteBuilder::new(Arc::new(dataset.clone()), predicate);
    run_delete(dataset, builder).await
}

/// Delete the rows that match `expr`, a DataFusion filter on the Lance
/// table columns.
///
/// Lance gets `expr` with no change. Unlike [`crate::NdScanner::filter_expr`],
/// a delete does not change value parts into stats filters, because a stats
/// filter keeps rows that do not match. Returns the number of deleted rows.
#[tracing::instrument(skip_all, fields(uri = %dataset.uri(), predicate = %expr), err)]
pub async fn delete_expr(dataset: &mut Dataset, expr: Expr) -> Result<u64> {
    check_format(dataset)?;
    let builder = DeleteBuilder::from_expr(Arc::new(dataset.clone()), expr);
    run_delete(dataset, builder).await
}

async fn run_delete(dataset: &mut Dataset, builder: DeleteBuilder) -> Result<u64> {
    let result = builder.execute().await?;
    *dataset = Arc::unwrap_or_clone(result.new_dataset);
    Ok(result.num_deleted_rows)
}

/// What a [`compact`] changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactionStats {
    /// Fragments that the compaction replaced.
    pub fragments_removed: usize,
    /// New fragments.
    pub fragments_added: usize,
    /// Data and deletion files that the new version no longer uses.
    pub files_removed: usize,
    /// New data files.
    pub files_added: usize,
}

/// What a [`cleanup_old_versions`] removed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CleanupStats {
    /// The bytes of the removed files.
    pub bytes_removed: u64,
    /// The number of removed versions.
    pub old_versions: u64,
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
    /// Merge fragments with similar variables from anywhere in the table,
    /// not only neighbors. The new fragments then hold few columns, which
    /// keeps reads and compactions of tables with many variables fast. The
    /// rows do not keep their order. The default is `false`.
    pub cluster: bool,
}

impl Default for CompactOptions {
    fn default() -> Self {
        Self {
            target_fragment_bytes: 128 << 20,
            deleted_fraction: 0.1,
            cluster: false,
        }
    }
}

/// Merge small fragments, split large fragments and remove deleted rows.
/// Keeps the row order, except with [`CompactOptions::cluster`].
///
/// The plan uses the size of each fragment on disk. Each run of
/// neighboring fragments that need work becomes fragments of about the
/// same size, near the target. A row is never split, so a row larger than
/// the target is a fragment on its own. lance-nd holds about one new
/// fragment in memory at a time. Without work to do, the table keeps its
/// version.
#[tracing::instrument(skip_all, fields(uri = %dataset.uri()), err)]
pub async fn compact(dataset: &mut Dataset, options: &CompactOptions) -> Result<CompactionStats> {
    check_format(dataset)?;
    let fragments: Vec<Fragment> = dataset.fragments().as_ref().clone();
    let mut pieces = Vec::with_capacity(fragments.len());
    for fragment in &fragments {
        pieces.push(piece_of(dataset, fragment).await?);
    }
    let target = options.target_fragment_bytes.max(1);
    let top_level = top_level_positions(dataset.schema());
    let columns: Vec<Vec<usize>> = fragments
        .iter()
        .map(|fragment| columns_of(fragment, &top_level))
        .collect();
    let tasks: Vec<Vec<usize>> = if options.cluster {
        cluster_plan(&pieces, &columns, target, options.deleted_fraction)
    } else {
        plan(&pieces, target, options.deleted_fraction)
            .into_iter()
            .map(|range| range.collect())
            .collect()
    };
    let mut metrics = CompactionStats::default();
    if tasks.is_empty() {
        return Ok(metrics);
    }
    let read_version = dataset.version().version;
    let mut bare_manifest = dataset.manifest.as_ref().clone();
    bare_manifest.schema = LanceSchema {
        fields: Vec::new(),
        metadata: Default::default(),
    };
    let mut groups = Vec::with_capacity(tasks.len());
    for task in tasks {
        let old: Vec<Fragment> = task.iter().map(|&i| fragments[i].clone()).collect();
        let task_pieces: Vec<Piece> = task.iter().map(|&i| pieces[i]).collect();
        let task_columns: Vec<Vec<usize>> = task.iter().map(|&i| columns[i].clone()).collect();
        let new_fragments = rewrite(
            dataset,
            &bare_manifest,
            &old,
            &task_pieces,
            &task_columns,
            target,
        )
        .await?;
        metrics.fragments_removed += old.len();
        metrics.files_removed += old
            .iter()
            .map(|f| f.files.len() + usize::from(f.deletion_file.is_some()))
            .sum::<usize>();
        metrics.fragments_added += new_fragments.len();
        metrics.files_added += new_fragments.len();
        groups.push(RewriteGroup {
            old_fragments: old,
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
    tracing::info!(
        version = dataset.version().version,
        fragments_removed = metrics.fragments_removed,
        fragments_added = metrics.fragments_added,
        "compacted the table"
    );
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
                dataset
                    .object_store(file.base_id)
                    .await?
                    .size(&path)
                    .await?
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
    let rewrite = |p: &Piece| needs_rewrite(p, target, deleted_fraction);
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

/// The most columns of a merged fragment in cluster mode, as a multiple of
/// its widest old fragment.
const CLUSTER_WIDTH: usize = 4;

/// The columns that a merged fragment can always hold in cluster mode.
const CLUSTER_MIN_COLUMNS: usize = 256;

/// The number of hashes in the similarity key of a fragment.
const SIGNATURE_HASHES: u64 = 4;

/// The groups of fragments to rewrite in cluster mode, as indexes.
///
/// The fragments that need work, as [`plan`] marks them, are sorted by a
/// MinHash signature of their columns, so that fragments with similar
/// variables come next to each other. Then each run of sorted fragments
/// becomes a group while its union of columns stays below
/// [`CLUSTER_WIDTH`] times its widest fragment, or [`CLUSTER_MIN_COLUMNS`],
/// and its live bytes stay below `target`. A group of one fragment that
/// needs no rewrite of its own stays.
fn cluster_plan(
    pieces: &[Piece],
    columns: &[Vec<usize>],
    target: u64,
    deleted_fraction: f32,
) -> Vec<Vec<usize>> {
    let small = |p: &Piece| p.bytes < target / 2;
    let rewrite = |p: &Piece| needs_rewrite(p, target, deleted_fraction);
    let mut candidates: Vec<(Vec<u64>, usize)> = (0..pieces.len())
        .filter(|&i| small(&pieces[i]) || rewrite(&pieces[i]))
        .map(|i| (signature(&columns[i]), i))
        .collect();
    candidates.sort_unstable();

    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut group: Vec<usize> = Vec::new();
    let mut union: HashSet<usize> = HashSet::new();
    let mut widest = 0;
    let mut bytes = 0.0;
    let close = |group: &mut Vec<usize>, groups: &mut Vec<Vec<usize>>| {
        if group.len() > 1 || group.first().is_some_and(|&i| rewrite(&pieces[i])) {
            group.sort_unstable();
            groups.push(std::mem::take(group));
        }
        group.clear();
    };
    for (_, i) in candidates {
        let piece = &pieces[i];
        let piece_bytes = piece.row_bytes() * piece.live_rows() as f64;
        let new = columns[i].iter().filter(|c| !union.contains(c)).count();
        let limit = (CLUSTER_WIDTH * widest.max(columns[i].len())).max(CLUSTER_MIN_COLUMNS);
        if !group.is_empty() && (union.len() + new > limit || bytes + piece_bytes > target as f64) {
            close(&mut group, &mut groups);
            union.clear();
            widest = 0;
            bytes = 0.0;
        }
        group.push(i);
        union.extend(columns[i].iter().copied());
        widest = widest.max(columns[i].len());
        bytes += piece_bytes;
    }
    close(&mut group, &mut groups);
    groups
}

/// A MinHash signature of `columns`: similar sets of columns get similar
/// signatures, so a sort puts them near each other.
fn signature(columns: &[usize]) -> Vec<u64> {
    (0..SIGNATURE_HASHES)
        .map(|seed| {
            columns
                .iter()
                .map(|&column| mix(column as u64 ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)))
                .min()
                .unwrap_or(u64::MAX)
        })
        .collect()
}

/// The SplitMix64 finalizer, a fast hash with a fixed result.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The top-level position in `schema` of each field id, also of nested fields.
fn top_level_positions(schema: &LanceSchema) -> HashMap<i32, usize> {
    fn visit(field: &lance_core::datatypes::Field, position: usize, map: &mut HashMap<i32, usize>) {
        map.insert(field.id, position);
        for child in &field.children {
            visit(child, position, map);
        }
    }
    let mut map = HashMap::new();
    for (position, field) in schema.fields.iter().enumerate() {
        visit(field, position, &mut map);
    }
    map
}

/// The sorted top-level positions of the columns that the data files of
/// `fragment` hold. A data file lists the ids of leaf fields, not always of
/// their parents. Ids of dropped fields are skipped.
fn columns_of(fragment: &Fragment, top_level: &HashMap<i32, usize>) -> Vec<usize> {
    let columns: HashSet<usize> = fragment
        .files
        .iter()
        .flat_map(|file| file.fields.iter())
        .filter_map(|id| top_level.get(id).copied())
        .collect();
    let mut columns: Vec<usize> = columns.into_iter().collect();
    columns.sort_unstable();
    columns
}

/// `schema` with only the top-level fields at `positions`.
fn project_positions(schema: &LanceSchema, positions: &[usize]) -> LanceSchema {
    LanceSchema {
        fields: positions
            .iter()
            .map(|&position| schema.fields[position].clone())
            .collect(),
        metadata: schema.metadata.clone(),
    }
}

/// `true` when a fragment needs a rewrite on its own: it is above twice
/// `target` with more than one row, or has at least `deleted_fraction` of
/// its rows deleted.
fn needs_rewrite(p: &Piece, target: u64, deleted_fraction: f32) -> bool {
    let large = p.bytes > target.saturating_mul(2) && p.rows > 1;
    let deleted = deleted_fraction <= 1.0
        && p.deleted > 0
        && p.deleted as f32 >= deleted_fraction * p.rows as f32;
    large || deleted
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
/// as a writer writes them. The others read as null. `columns` has the
/// top-level positions of the columns of each fragment: a fragment is read
/// with only its own columns, as a read of all columns takes a time that
/// grows with the columns of the table. `manifest` is the manifest of
/// `dataset` without its schema.
async fn rewrite(
    dataset: &Dataset,
    manifest: &Manifest,
    fragments: &[Fragment],
    pieces: &[Piece],
    columns: &[Vec<usize>],
    target: u64,
) -> Result<Vec<Fragment>> {
    let mut union: Vec<usize> = columns
        .iter()
        .flatten()
        .copied()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    union.sort_unstable();
    let lance_schema = project_positions(dataset.schema(), &union);
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
    // Float sums miss an exact cut by a tiny amount, which would add a row.
    let slack = part_bytes * 1e-9;
    let mut cuts = 1;
    let mut written = 0.0;
    let mut pending: Vec<RecordBatch> = Vec::new();
    let mut new_fragments = Vec::new();
    // Lance looks up each projected struct of a fragment read in the table
    // schema, in a time linear in its columns. A narrow schema avoids that.
    let mut narrow_manifest = manifest.clone();
    narrow_manifest.schema = writer.lance_schema.clone();
    let mut narrow = dataset.clone();
    narrow.manifest = Arc::new(narrow_manifest);
    let shared = Arc::new(narrow);
    for ((fragment, piece), own) in fragments.iter().zip(pieces).zip(columns) {
        let row_bytes = piece.row_bytes().max(f64::MIN_POSITIVE);
        // A fragment reader, not a scan: a scan plans with the full schema,
        // which takes a time linear in the table columns for each fragment.
        let reader = FileFragment::new(shared.clone(), fragment.clone())
            .open(
                &project_positions(dataset.schema(), own),
                FragReadConfig::default(),
            )
            .await?;
        // Small batches, so that a cut can fall between them.
        let mut stream = reader.read_all(64).await?.buffered(2);
        while let Some(batch) = stream.try_next().await? {
            let mut batch = writer.conform(&batch)?;
            while batch.num_rows() > 0 {
                let take = if cuts < count {
                    let left = part_bytes * cuts as f64 - written - slack;
                    ((left / row_bytes).ceil() as usize).clamp(1, batch.num_rows())
                } else {
                    batch.num_rows()
                };
                pending.push(batch.slice(0, take));
                written += row_bytes * take as f64;
                batch = batch.slice(take, batch.num_rows() - take);
                if cuts < count && written + slack >= part_bytes * cuts as f64 {
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
            store_params: self.dataset.store_params().cloned(),
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

/// Remove the files of the table versions older than `older_than`. The
/// current version always stays.
///
/// A reader that still uses a removed version fails, so keep a margin for
/// long scans.
#[tracing::instrument(skip_all, fields(uri = %dataset.uri(), older_than = ?older_than), err)]
pub async fn cleanup_old_versions(dataset: &Dataset, older_than: Duration) -> Result<CleanupStats> {
    let older_than = chrono::Duration::from_std(older_than).unwrap_or(chrono::Duration::MAX);
    let removed = dataset.cleanup_old_versions(older_than, None, None).await?;
    Ok(CleanupStats {
        bytes_removed: removed.bytes_removed,
        old_versions: removed.old_versions,
    })
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
        batch
            .column(0)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec()
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
