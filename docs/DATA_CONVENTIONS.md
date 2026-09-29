# Data conventions and input contracts

## Axes, units, and field of view

| Representation | Coordinate order | Units / sampling |
| --- | --- | --- |
| NumPy / TIFF volume | Z, Y, X | Explicitly record voxel spacing |
| CloudVolume indexing | X, Y, Z | Use its scale resolution and voxel offset |
| Released raw crop | Z, Y, X | 50, 16, 16 nm |
| Native mitochondrial / ER masks | Z, Y, X | 50, 8, 8 nm |
| Native Golgi instance masks | Z, Y, X | 50, 16, 16 nm |
| Contact computation | Z, Y, X | 50, 4, 4 nm |
| Legacy mitochondrial OBJ vertex | Y, X, Z | Already in nm; not voxel indices |

For a legacy OBJ vertex `v = (Y_nm, X_nm, Z_nm)`, conventional XYZ is `v[[1, 0, 2]]`. With a raw crop origin `o = (z0, y0, x0)` in the source 16-nm grid, the corresponding continuous crop-array coordinate is:

```python
crop_zyx = v[[2, 0, 1]] / [50.0, 16.0, 16.0] - [z0, y0, x0]
```

Use sidecar origins rather than assuming meshes are centred in a crop. Meshes outside the crop are expected. The source CloudVolume offsets were zero for this prepared dataset; a different dataset's offsets must be applied explicitly.

Nearest-neighbour mask replication preserves native voxel coverage and the shared origin. Do not resize one field of view to another field's dimensions or infer unobserved edge strips. The distance kernel uses occupied grid coordinates, not continuous fitted membrane surfaces. The selected coordinate convention is part of the operational measurement.

## Contact input hierarchy

The `--root` argument is a project root with this study-specific layout:

```text
PROJECT_ROOT/
├── result/
│   ├── cell1_masked_test/mito_instance/snitched_stacks/
│   ├── cell2_masked_test/mito_instance/snitched_stacks/
│   └── cell3_masked_test/mito_instance/mito_after_proof/
└── dist_result_2026/data/
    ├── cell1/{er/,golgi/golgi_instance/}
    ├── cell2/{er/,golgi/golgi_instance/}
    └── cell3/{er/,golgi/golgi_instance/}
```

`snitched_stacks` is the original directory spelling, not an instruction to rename source data. Each directory contains single-section PNG/TIFF images. The first number in each filename supplies the section index: mitochondrial/ER sections are 1-based; Golgi filenames are 0-based and receive an offset of +1. There must be one image for every expected section. Avoid filenames whose first number is a cell identifier instead of the section number.

| Cell | Native mitochondrial / ER ZYX shape | Native Golgi ZYX shape |
| --- | --- | --- |
| 1 | 544 × 6625 × 8480 | 544 × 3312 × 4240 |
| 2 | 340 × 4500 × 5000 | 340 × 2250 × 2500 |
| 3 | 415 × 4560 × 4728 | 415 × 2280 × 2364 |

Mitochondrial images contain nonnegative integer instance IDs, with zero as background. ER must be encoded as 0/1. Golgi uses nonzero integer IDs or palette indices as foreground; do not substitute RGB preview images. Cell 1's shorter Golgi field leaves two computation-grid rows unobserved; affected mitochondrial halos are flagged. The flag is not evidence that these locations were experimentally observed to be negative.

### Cohort CSV (`--cohort`)

Required fields:

```csv
cell,instance_id,morphology
Cell 1,3,compact
```

This is a schema illustration, not a supplied result row. The first-stage reader expects `Cell 1`, `Cell 2`, or `Cell 3`, not integer cell values. It retains the supplied instance IDs and morphology labels; it does not infer morphology or perform classification. Provide exactly 2676 / 530 / 624 unique IDs for the respective cells. This illustrative header is not a substitute for the real cohort table.

### Bounds CSV (`--bounds`)

Required fields:

```text
cell,role,instance_id,min_z,max_z_excl,min_r,max_r_excl,min_c,max_c_excl,voxel_count
```

`cell` is an integer. Use `role=native_stitched` for Cells 1/2 and `role=native_proofread` for Cell 3. Bounds are **zero-based, end-exclusive** on the original 50×8×8-nm label grid; `r` means Y and `c` means X. `voxel_count` is the native-grid count, not the count after replication. The first-stage script checks archived, current, and cropped counts for equality. The optional historical `touches_outer_face` column is tolerated. Do not add arbitrary nonnumeric columns to this study-specific table reader.

## Contact output contracts

The first stage writes a `cellN/` folder with:

- `configuration.json`: spacing, thresholds, cohort hashes, source paths, and mapping settings.
- `input_manifest.csv`: decoded per-section hashes and image metadata.
- `cohort_count_audit.csv`: native object-count agreement.
- `per_instance_proximity.csv`: one row per ID × partner, with `measurement_protocol=as_provided_labels_v1`.
- `threshold_summary.csv`: positive counts, fixed denominators, and percentages.
- `COMPLETE.json`: `status=PASS` only after the script's checks succeed.

`distance_status=resolved_le100_nm` has an exact `min_distance_nm_le100`. `greater_than_100_nm` has an empty numerical distance. Threshold columns are binary `le_30nm`, `le_50nm`, `le_80nm`, and `le_100nm`, using an inclusive comparison and the code's floating-point tolerance.

`assemble_primary_table.py` pairs ER/Golgi rows by cell and instance ID. Its output has integer `cell`, `instance_id`, `morphology`, `measurement_protocol`, and partner-prefixed fields such as `er_distance_status`, `er_distance_nm_le100`, and `er_le_30nm`. It preserves censoring and flags and refuses duplicate, incomplete, incompatible, or inconsistent object pairs. This is the `--primary` input for the extension. Both stages require certified prior outputs carrying the current measurement protocol; earlier run schemas must not be reused or relabelled as current results.

The full-range extension writes `per_instance_full_distances.csv` with `er_min_distance_nm` and `golgi_min_distance_nm`, method labels, independent witness checks, and a completion report. It verifies the decoded source hashes against the first-stage manifest and checks that all threshold calls remain unchanged. Its full-target KD-tree can require substantial RAM. Choose worker counts for the actual server; this is not a streaming constant-memory implementation.

Use a new output directory for each real run. Completed outputs are not overwritten. The extension creates a `RUNNING.lock` to guard duplicate launches; after failure, inspect logs and process state before deciding how to restart. Do not remove a lock belonging to an active process.

## Point-cloud HDF5

`cloudpoints`: float array `(objects, points, 3)` with at least 1024 finite points per object. The release mesh utility defaults to 4096 samples and also writes UTF-8 `source_mesh` filenames, a random seed, and normalization metadata. These normalized coordinates are dimensionless and do not preserve physical organelle size.

The training loader's final-object exclusion follows HDF5 row order. Changing `holdout_last_n` or the source ordering changes the included objects; document it. The example utility is not a replacement for the historical training-data provenance.
