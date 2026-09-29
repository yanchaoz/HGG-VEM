# Mitochondria-ER / mitochondria-Golgi proximity

These are the actual revised analysis implementations, not the legacy mesh-distance routines.

- Computation grid, **ZYX: (50, 4, 4) nm**. Distances are Euclidean distances between occupied voxel centres; they are not fitted continuous membrane-to-membrane distances.
- Native mitochondrial / ER inputs: 8-nm XY; Golgi instance input: 16-nm XY. Use integer nearest-neighbour replication with preserved origin and field of view, never stretch images to fit. Golgi foreground is `golgi_instance != 0`.
- Each mitochondrial ID is dilated **independently** with an XY disk of radius 4 computation-grid pixels (16 nm). No Z dilation. IDs are never merged by this dilation.
- Paired radius-zero control is retained. Threshold calls use **<= 30 / 50 / 80 / 100 nm**.
- Fixed cohorts: 2676 / 530 / 624. Cell 1's incomplete outer Golgi strip is flagged; unknown regions are not inferred from an image resize.
- `recompute_mask_proximity.py` guarantees threshold-range distances only. Values >100 nm are censored in that stage, not fabricated maxima. `extend_fullrange_proximity.py` extends the distance calculation with a certified nearest-block search.

Test numerical kernels without private input data:

```bash
python recompute_mask_proximity.py --self-test
python extend_fullrange_proximity.py --self-test
```

For full-volume use, supply `--root`, `--cohort`, `--bounds`, `--cell`, and `--output`; the extension additionally requires the preceding run and primary table. See `sources()` for the expected source hierarchy. The fixed acquisition dimensions and cohort checks deliberately guard this study's inputs; using another dataset requires an explicit configuration adaptation.

The public 16-nm raw crops and meshes do not include the 4-nm-grid partner masks needed to rerun this complete analysis. This module exposes the method; it does not imply that all supporting input volumes are in this release.

## Full-volume workflow

These commands use placeholders and require the actual study-specific inputs described in [DATA_CONVENTIONS.md](../../docs/DATA_CONVENTIONS.md). Run from this module directory. Replace paths before use. The data needed for these commands are not included in the raw-crop/mesh example package.

```bash
# 1. Check section counts, cohort IDs, bounds, and expected source hierarchy.
python recompute_mask_proximity.py \
  --root /path/to/PROJECT_ROOT --cohort /path/to/cohort.csv \
  --bounds /path/to/bounds.csv --cell 1 --workers 2 \
  --output /path/to/new_threshold_run --check-only

# 2. Calculate threshold-range distances and the radius-zero control.
python recompute_mask_proximity.py \
  --root /path/to/PROJECT_ROOT --cohort /path/to/cohort.csv \
  --bounds /path/to/bounds.csv --cell 1 --workers 2 \
  --output /path/to/new_threshold_run

# 3. Pair primary radius-4 ER and Golgi records; keep censored distances empty.
python assemble_primary_table.py \
  --previous /path/to/new_threshold_run --cell 1 \
  --output /path/to/cell1_primary.csv

# 4. Optional: resolve the full distance range; this can be memory-intensive.
python extend_fullrange_proximity.py \
  --root /path/to/PROJECT_ROOT --previous /path/to/new_threshold_run \
  --primary /path/to/cell1_primary.csv --bounds /path/to/bounds.csv \
  --cell 1 --workers 2 --output /path/to/new_fullrange_run
```

Repeat with cells 2 and 3 and distinct primary CSV filenames. Both run directories create per-cell subdirectories. `--check-only` validates file indexing and cohort/bounds correspondence but does not read every segmentation pixel; the real run additionally checks image shapes and native voxel counts.

`assemble_primary_table.py` is a new release adapter, not a change to either distance kernel. It enforces the fixed cohort, complete ER/Golgi pairs, morphology consistency, and consistency between distances and threshold calls. Its output schema is the wide table consumed by the full-range extension.
