# Organelle proximity from input labels

Compute mitochondrial distances to ER and Golgi foreground directly from the supplied segmentation labels.

## Measurement

- Computation grid: **ZYX = 50 × 4 × 4 nm**.
- Native mitochondrial/ER labels: 8-nm XY; Golgi instance labels: 16-nm XY.
- Map labels by integer nearest-neighbour replication, preserving the origin and field of view. Do not stretch one image to fit another.
- Mitochondrial IDs retain their supplied geometry. ER foreground is 1; Golgi foreground is any nonzero instance ID.
- Minimum Euclidean distances are between occupied voxel centres, not fitted continuous membrane surfaces.
- Threshold calls use **≤ 30 / 50 / 80 / 100 nm**. The denominator is the fixed cohort: 2676 / 530 / 624.
- Cell 1's incomplete outer Golgi strip is flagged; unobserved locations are not reconstructed by resizing.

The binary occurrence statistic is the fraction of cohort mitochondria with a minimum distance at or below the threshold. It is not a contact count or a surface-area-weighted measurement.

## Scripts

- `recompute_mask_proximity.py`: exact distances through 100 nm and threshold calls. Larger distances remain censored, with an empty numerical value.
- `assemble_primary_table.py`: pair ER/Golgi rows into the wide input required by the next stage; preserve censoring and quality-control flags.
- `extend_fullrange_proximity.py`: resolve the remaining distances with a certified nearest-block search and check that threshold calls are unchanged.

Outputs carry `measurement_protocol=as_provided_labels_v1`. The table adapter and extension reject incompatible or incomplete previous runs. Use newly generated threshold outputs with this version.

## Numerical checks

```bash
python recompute_mask_proximity.py --self-test
python extend_fullrange_proximity.py --self-test
```

## Full-volume workflow

Run from this module directory. Replace the example paths with the registered inputs described in [DATA_CONVENTIONS.md](../../docs/DATA_CONVENTIONS.md). The representative raw-crop/mesh package does not contain all masks and tables required for full-volume analysis.

```bash
# Validate section indexing and cohort/bounds correspondence.
python recompute_mask_proximity.py \
  --root /path/to/PROJECT_ROOT --cohort /path/to/cohort.csv \
  --bounds /path/to/bounds.csv --cell 1 --workers 2 \
  --output /path/to/new_label_distance_run --check-only

# Compute threshold-range results.
python recompute_mask_proximity.py \
  --root /path/to/PROJECT_ROOT --cohort /path/to/cohort.csv \
  --bounds /path/to/bounds.csv --cell 1 --workers 2 \
  --output /path/to/new_label_distance_run

# Pair the ER and Golgi records.
python assemble_primary_table.py \
  --previous /path/to/new_label_distance_run --cell 1 \
  --output /path/to/cell1_primary.csv

# Optional: resolve full-range distances.
python extend_fullrange_proximity.py \
  --root /path/to/PROJECT_ROOT --previous /path/to/new_label_distance_run \
  --primary /path/to/cell1_primary.csv --bounds /path/to/bounds.csv \
  --cell 1 --workers 2 --output /path/to/new_fullrange_run
```

Repeat for Cells 2 and 3, with distinct primary CSV filenames. Output roots contain separate `cellN/` directories. The real run checks image shapes and native voxel counts in addition to the indexing checks.

The full-range stage uses boundary voxels from the unmodified mitochondrial support as query points. Only distances greater than 100 nm reach this search, so their nearest source voxel lies on that boundary. This point-selection optimization does not replace or alter the input mask. The full-target KD-tree can require substantial memory.

Study-specific dimensions and cohort checks are intentional. Adapting the code to another acquisition requires explicit changes to the input configuration and corresponding validation.
