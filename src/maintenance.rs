//! Delete rows, compact the table and clean up old versions.
//!
//! A delete marks rows in deletion files and keeps the data files. A
//! compaction rewrites small fragments and fragments with many deleted rows.
//! Both make a new table version. The files of old versions stay on disk
//! until [`cleanup_old_versions`] removes them.

use std::collections::HashSet;
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
use lance::dataset::optimize::{CompactionMetrics, CompactionOptions, plan_compaction};
use lance::dataset::statistics::DatasetStatisticsExt;
use lance::dataset::transaction::{Operation, RewriteGroup, Transaction};
use lance::dataset::write::delete::DeleteBuilder;
use lance::dataset::{CommitBuilder, WriteMode, WriteParams};
use lance::deps::datafusion::logical_expr::Expr;
use lance::table::format::Fragment;

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
    /// A fragment smaller than about this many bytes on disk is merged with
    /// its neighbors. The default is 128 MiB.
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

/// Merge small fragments and remove deleted rows. Keeps the row order.
///
/// Lance plans the work. Lance selects fragments by rows, so the target in
/// bytes becomes a target in rows from the mean row size on disk. lance-nd
/// rewrites each group of fragments as one fragment, and holds the group in
/// memory. A group with more list or string values than one Arrow array
/// holds becomes several fragments. Without work to do, the table keeps its
/// version.
pub async fn compact(dataset: &mut Dataset, options: &CompactOptions) -> Result<CompactionMetrics> {
    let plan = plan_compaction(dataset, &lance_options(dataset, options).await?).await?;
    let mut metrics = CompactionMetrics::default();
    if plan.tasks.is_empty() {
        return Ok(metrics);
    }
    let mut groups = Vec::with_capacity(plan.tasks.len());
    for task in &plan.tasks {
        let new_fragments = rewrite(dataset, &task.fragments).await?;
        metrics.fragments_removed += task.fragments.len();
        metrics.files_removed += task
            .fragments
            .iter()
            .map(|f| f.files.len() + usize::from(f.deletion_file.is_some()))
            .sum::<usize>();
        metrics.fragments_added += new_fragments.len();
        metrics.files_added += new_fragments.len();
        groups.push(RewriteGroup {
            old_fragments: task.fragments.clone(),
            new_fragments,
        });
    }
    // New fragments need ids that no other writer takes.
    let count: usize = groups.iter().map(|g| g.new_fragments.len()).sum();
    let reserve = Operation::ReserveFragments {
        num_fragments: count as u32,
    };
    let reserved = CommitBuilder::new(Arc::new(dataset.clone()))
        .execute(Transaction::new(plan.read_version, reserve, None))
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
        .execute(Transaction::new(plan.read_version, rewrite, None))
        .await?;
    Ok(metrics)
}

/// The rows of `fragments`, written again as new fragments without ids.
///
/// The new fragments hold only the top-level columns that `fragments` hold,
/// as a writer writes them. The others read as null.
async fn rewrite(dataset: &Dataset, fragments: &[Fragment]) -> Result<Vec<Fragment>> {
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
    let lance_schema = &dataset.schema().project(&names)?;
    let arrow_schema: &SchemaRef = &Arc::new(Schema::from(lance_schema));
    let mut scanner = dataset.scan();
    scanner.project(&names)?;
    scanner.with_fragments(fragments.to_vec());
    let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
    let batches = batches
        .iter()
        .map(|batch| {
            let batch = conform_batch(batch, arrow_schema)?;
            let columns = arrow_schema
                .fields()
                .iter()
                .zip(batch.columns())
                .map(|(field, column)| fill_hidden_nulls(column, field))
                .collect::<Result<Vec<_>>>()?;
            Ok(RecordBatch::try_new(arrow_schema.clone(), columns)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let params = WriteParams {
        mode: WriteMode::Append,
        data_storage_version: Some(FILE_VERSION),
        ..Default::default()
    };
    let concat = |part: &[RecordBatch]| concat_batches(arrow_schema, part);
    let mut new_fragments = Vec::new();
    // One batch per fragment: Lance 11 fails to read back a fragment of
    // several batches when a struct column has nulls and large lists.
    for batch in split_concat(&batches, &concat)? {
        if batch.num_rows() == 0 {
            continue;
        }
        let reader = RecordBatchIterator::new([Ok(batch)], arrow_schema.clone());
        new_fragments.extend(
            FragmentCreateBuilder::new(dataset.uri())
                .schema(lance_schema)
                .write_params(&params)
                .write_fragments(reader)
                .await?,
        );
    }
    Ok(new_fragments)
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

/// The Lance compaction options for `options`.
async fn lance_options(dataset: &Dataset, options: &CompactOptions) -> Result<CompactionOptions> {
    let rows = dataset.count_rows(None).await? as u64;
    let bytes: u64 = Arc::new(dataset.clone())
        .calculate_data_stats()
        .await?
        .fields
        .iter()
        .map(|field| field.bytes_on_disk)
        .sum();
    let row_bytes = (bytes / rows.max(1)).max(1);
    let lance_options = CompactionOptions {
        target_rows_per_fragment: (options.target_fragment_bytes / row_bytes).max(1) as usize,
        materialize_deletions: options.deleted_fraction <= 1.0,
        materialize_deletions_threshold: options.deleted_fraction,
        ..Default::default()
    };
    Ok(lance_options)
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
