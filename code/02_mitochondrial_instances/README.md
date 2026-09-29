# Foreground + contour to mitochondrial instances

`connectomics/process.py` and `misc.py` preserve the supplied historical post-processing kernels and their MitoEM attribution. `instances_from_probabilities.py` is a new command-line adapter.

```bash
python instances_from_probabilities.py --foreground foreground.tif --boundary contour.tif --output instances.tif
```

Inputs are co-registered 3D ZYX volumes: float probabilities in [0,1], or uint8 in [0,255]. The default connected-component method uses the original `bc_connected` kernel with foreground threshold 0.65 and contour threshold 0.50 (the manuscript parameter preset), XY dilation shape (1,5,5), and minimum volume 128 voxels. The retained kernel quantizes threshold values on the uint8 scale. Connected components use the kernel's full connectivity; its label-valued dilation resolves collisions by maximum label. These choices are explicit and are not a physical-space nearest-label expansion.

The wrapper runs on one assembled volume. It is **not** a reimplementation of distributed block stitching. Align and blend network probability outputs before decoding; do not concatenate independently numbered instances as if their IDs were globally unique. The historical driver files contain mixed cell paths and experimental alternatives, so they are not exposed as ready-to-run production commands.

`--method validation-watershed` selects the separate later probability-boundary audit method: foreground support = interior + boundary >=0.50; seeds = interior >=0.50 outside a boundary >=0.35 barrier dilated in XY. Here `--foreground` means the **interior-class probability**, not an already combined foreground map. Barrier expansion changes seeds only, not the support mask. This method is not silently substituted for the connected-component production preset.

Optional `--body-mask` clips outside-cell voxels; optional `--minimum-max-slice-area 128` requires at least one XY section with 128 pixels for each retained object. This is different from minimum total object volume, and its physical area depends on the actual XY sampling. No body clipping or slice-area filter is applied unless requested.

The 16-nm expansion used in the contact module is a separate per-instance distance-analysis operation; do not conflate it with this module's instance reconstruction steps. This release adapter has synthetic tests, not a new production reconstruction or a claim of reproduced instance accuracy.
