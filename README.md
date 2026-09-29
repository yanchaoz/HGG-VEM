# HGG-VEM

Core computational methods for volume electron microscopy of high-grade glioma.

HGG-VEM brings together masked image pretraining, mitochondrial instance reconstruction, inter-organelle proximity analysis, and self-supervised learning of mitochondrial point-cloud representations. The repository exposes the methods used in the study together with explicit input conventions, runnable numerical checks, and practical examples.

**Scope.** This is a research code release, not an end-to-end reproduction of every manuscript experiment. It does not include the mitochondrial volume/surface-area/length measurement pipeline, pretrained weights, complete training datasets, reference annotations, or ER/Golgi segmentation volumes. Raw EM examples and mitochondrial meshes are managed separately from Git.

## Contents

- [Repository structure](#repository-structure)
- [Installation and quick checks](#installation-and-quick-checks)
- [1. Masked image modeling](#1-masked-image-modeling)
- [2. Mitochondrial instance reconstruction](#2-mitochondrial-instance-reconstruction)
- [3. Mitochondria-ER and mitochondria-Golgi proximity](#3-mitochondria-er-and-mitochondria-golgi-proximity)
- [4. Mitochondrial point-cloud SimSiam](#4-mitochondrial-point-cloud-simsiam)
- [Associated data and coordinates](#associated-data-and-coordinates)
- [Validation and reproducibility](#validation-and-reproducibility)
- [Acknowledgements and licensing](#acknowledgements-and-licensing)

## Repository structure

```text
HGG-VEM/
├── code/
│   ├── 01_masked_image_modeling/    # SparK-style pretraining; default 3D U-Net
│   ├── 02_mitochondrial_instances/ # Foreground + boundary probability decoding
│   ├── 03_contact_analysis/        # Threshold and full-range voxel-grid distances
│   └── 04_pointcloud_ssl/          # PointNet++ encoder and SimSiam training
├── docs/
│   ├── DATA_CONVENTIONS.md         # Grids, file contracts, and coordinate transforms
│   ├── VALIDATION.md               # Tests, limitations, and reproducibility boundary
│   ├── PACKAGING_CHANGES.md        # Differences from the supplied research scripts
│   └── TEST_REPORT.json            # Recorded pre-release CPU checks
├── tests/                         # Synthetic tests; no patient data required
└── THIRD_PARTY_NOTICES.md          # Upstream acknowledgements and licence references
```

Each module has its own README and `requirements.txt`. Run module-specific commands **from that module's directory** unless an example explicitly starts at the repository root. Examples below use Bash and placeholder input paths; replace these with your own co-registered inputs.

## Installation and quick checks

Clone the repository:

```bash
git clone https://github.com/yanchaoz/HGG-VEM.git
cd HGG-VEM
```

The numerical analysis and instance post-processing checks do not require a GPU or private data:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r code/02_mitochondrial_instances/requirements.txt
python -m pip install -r code/03_contact_analysis/requirements.txt
python -m unittest discover -s tests -p 'test_numerical.py' -v
```

On Windows, activate with `.venv\Scripts\Activate.ps1` and use the corresponding PowerShell command syntax. Large-volume processing and the historical training scripts were developed on Linux.

Use **separate environments for the training modules**. Install a PyTorch/torchvision pair appropriate for your CUDA installation, then install the selected module's requirements. The historical training code uses `attrdict` and, for MIM, `timm==0.6.13` and **nnU-Net v1** (`nnunet==1.7.1`). These are not interchangeable with nnU-Net v2. A Python 3.8-era environment is the compatibility baseline for these legacy modules; an arbitrary current Python/PyTorch combination is not validated. Requirements files describe dependencies, not a fully locked historical environment.

Training commands below are entry-point examples, **not claims that GPU training was rerun for this release**. See [validation details](docs/VALIDATION.md).

## 1. Masked image modeling

[Module documentation](code/01_masked_image_modeling/README.md)

The default backbone is the supplied **3D U-Net / nnU-Net encoder**, constructed through `nnUNet_head.py::Generic_UNet`. `SparseEncoder`, `LightDecoder`, and `SparK` implement sparse masked reconstruction. The selected entry point is `pretrain_nn.py`; the retained ancillary STU-Net definitions do not change the default backbone.

Key settings in the actual training entry are:

| Setting | Value |
| --- | --- |
| Input | One-channel grayscale, 128 × 128 × 128 voxels |
| Mask ratio | 0.75 |
| Intensity normalization | uint8 intensity divided by 255 |
| Objective | Mean squared reconstruction error on masked patches |
| Checkpoint | `learner.ckpt` inside the configured run directory |

The legacy `TRAIN.loss_func` YAML field does not select the implemented MIM objective; the active objective is defined in `pretrain_nn.py`.

Arrange TIFF stacks in subdirectories and set `DATA.data_folder` in `config/Base_MAE.yaml`:

```text
raw_em_16nm_nucleus_centered/
├── cell1/raw_zyx_50_16_16nm.tif
├── cell2/raw_zyx_50_16_16nm.tif
└── cell3/raw_zyx_50_16_16nm.tif
```

```bash
cd code/01_masked_image_modeling
python -m pip install -r requirements.txt
# Edit config/Base_MAE.yaml before starting a real training run.
CUDA_VISIBLE_DEVICES=0 python pretrain_nn.py -c Base_MAE
```

The loader removes 12 pixels from each XY edge, so inputs need Z ≥ 128 and Y/X ≥ 152. It ignores JSON metadata and PNG previews. It currently reads a complete selected TIFF before sampling a crop: account for volume size and worker count when setting `TRAIN.num_workers`; reduce it for large example volumes. The released representative crops are examples, not the original pretraining corpus.

## 2. Mitochondrial instance reconstruction

[Module documentation](code/02_mitochondrial_instances/README.md)

This module decodes aligned **foreground/interior and boundary probabilities** into integer instance labels. It begins with network probability outputs; it does not train a segmentation network, run inference, or stitch independently labelled blocks.

Inputs must have identical **ZYX** shapes. Float probabilities must be finite and in [0, 1]; uint8 probabilities are interpreted on [0, 255]. `.tif`, `.tiff`, and `.npy` probability inputs are supported. Background has label 0 in the output.

```bash
cd code/02_mitochondrial_instances
python instances_from_probabilities.py \
  --foreground /path/to/foreground.tif \
  --boundary /path/to/boundary.tif \
  --output /path/to/new_output/instances.tif
```

The default `connected` method preserves the historical `bc_connected` kernel: foreground threshold 0.65, boundary threshold 0.50, XY dilation footprint `(1, 5, 5)`, and minimum object size 128 voxels. Thresholds are applied on the kernel's uint8 scale. Its label-valued dilation resolves collisions by maximum label; it is not a physical-space nearest-label expansion.

The separate validation configuration uses probability boundaries to define seeds:

```bash
python instances_from_probabilities.py \
  --method validation-watershed \
  --foreground /path/to/interior_probability.tif \
  --boundary /path/to/boundary_probability.tif \
  --seed-radius-xy 1 \
  --minimum-object-voxels 128 \
  --output /path/to/new_output/validation_instances.tif
```

For this method, `--foreground` specifically means **interior-class probability**. Support is `interior + boundary >= 0.50`; seeds use `interior >= 0.50` outside a `boundary >= 0.35` barrier. Expanding that barrier changes seeds only, not the foreground support. The connected-method threshold flags do not tune these fixed validation thresholds.

Optional `--body-mask` excludes outside-cell pixels. Optional `--minimum-max-slice-area 128` requires an object to occupy at least 128 pixels on one XY section; this is different from a 128-voxel total-volume filter. Neither option is enabled implicitly. Output includes a uint32 TIFF and JSON settings sidecar. Existing TIFF outputs are refused.

Reconcile spatial registration and overlapping probability predictions **before** instance decoding. Concatenating independently numbered instance masks is not valid seam handling.

## 3. Mitochondria-ER and mitochondria-Golgi proximity

[Module documentation and full-volume commands](code/03_contact_analysis/README.md)

The revised analysis operates on segmentation masks, not the historical mesh-distance tables. For each mitochondrial instance and partner organelle, it computes the minimum Euclidean distance between occupied voxel centres after the specified mapping and dilation.

| Quantity | Convention |
| --- | --- |
| Computation grid | **ZYX = 50 × 4 × 4 nm** |
| Native mitochondrial / ER labels | ZYX = 50 × 8 × 8 nm |
| Native Golgi labels | ZYX = 50 × 16 × 16 nm; nonzero instance IDs define foreground |
| Mapping | Integer nearest-neighbour replication in XY; preserve origin and field of view |
| Primary mitochondrial expansion | Independent XY disk, radius 4 computation-grid pixels = 16 nm |
| Z expansion | None |
| Control | No mitochondrial expansion |
| Thresholds | **≤ 30, 50, 80, and 100 nm** |
| Fixed cell cohorts | 2676 / 530 / 624 mitochondrial instances |

Each instance is expanded independently: overlapping expanded supports do not merge mitochondrial identities. Nearest-neighbour mapping to 4 nm does **not** create new native boundary information.

The occurrence statistic at threshold `t` is:

```text
number of cohort mitochondria with minimum distance <= t
------------------------------------------------------
              number of cohort mitochondria
```

It is a binary per-mitochondrion call, not contact-surface area, contact count, or surface-area-weighted proximity. Morphologically defined putative membrane contact sites (MCS) remain an operational proximity measure, not functional confirmation of membrane coupling.

`recompute_mask_proximity.py` returns exact threshold-range results through 100 nm. Larger values are represented as `greater_than_100_nm` with an empty numerical distance, **not** as invented maximum distances. `assemble_primary_table.py` provides an explicit, checked conversion of the expanded-instance results into the wide input required by `extend_fullrange_proximity.py`. The extension resolves the full range and verifies that the threshold calls do not change.

Both numerical kernels can be checked independently:

```bash
cd code/03_contact_analysis
python recompute_mask_proximity.py --self-test
python extend_fullrange_proximity.py --self-test
```

The full-volume entry points deliberately enforce study-specific cell dimensions and cohort checks. They require the original registered masks, cohort table, and instance bounds; these are not supplied by the representative raw-image/mesh release. See [input contracts](docs/DATA_CONVENTIONS.md) before adapting the scripts to another dataset. Cell 1's unobserved outer Golgi strip is retained as an explicit quality-control flag, not filled by stretching the segmentation.

## 4. Mitochondrial point-cloud SimSiam

[Module documentation](code/04_pointcloud_ssl/README.md)

The encoder is **PointNet++ / PointNetV2**, producing 1024-dimensional representations. The learning objective is **SimSiam**: two independently augmented views, a shared encoder, a projection/prediction head, and a stop-gradient target. There is **no EMA/momentum target encoder**. The supplied research implementation used a BYOL class name with momentum disabled; the release naming reflects that active algorithm.

The projection dimension is 128 and the hidden width is 512. The original two-direction normalized squared-distance loss, `2 - 2*cosine_similarity`, and its summation scale are preserved.

Prepare an HDF5 file containing `cloudpoints` with shape `(objects, points, 3)`. A release utility can sample the associated OBJ meshes:

```bash
cd code/04_pointcloud_ssl
python -m pip install -r requirements.txt
python mesh_to_pointcloud.py \
  --mesh-root /path/to/extracted_meshes \
  --output ./data/mitochondrial_pointclouds.h5 \
  --points 4096 --seed 20260929
```

The utility samples triangles in proportion to surface area, subtracts the point-cloud centroid, and scales to unit maximum radius. It saves source mesh filenames alongside the points. This is a new input-preparation utility, **not** a reconstruction of the exact historical training corpus; normalized clouds must not be reused as physical-coordinate morphometry data.

Set `DATA.pointcloud_h5` in `config/hgg_simsiam.yaml`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 python train_simsiam.py -c hgg_simsiam
```

The loader independently augments each view, farthest-point samples 1024 points, and normalizes it. `DATA.holdout_last_n: 16` preserves the historical exclusion of the final 16 input objects. This is an input-order exclusion, not a patient-independent evaluation split. It also applies to embedding extraction; use an explicit value of 0 if you intend to process every object. Keep `TRAIN.resume: false`: the historical checkpoints do not contain a complete optimizer/projector/predictor resume state.

Encoder checkpoints are written beneath `TRAIN.save_path/<timestamp>_<NAME>/`. For the existing extraction entry point, place a matching encoder checkpoint at `trained_model/<model_name>/<model_id>.ckpt`:

```bash
CUDA_VISIBLE_DEVICES=0 python extract_embeddings.py \
  -c hgg_simsiam -mn hgg_simsiam -id model-080000
```

This example expects `trained_model/hgg_simsiam/model-080000.ckpt`. The checkpoint must contain the `model_weights` dictionary for the same PointNetV2 encoder. The script writes `inference/hgg_simsiam/feat_model-080000/features.npy`, shaped `(objects, 1, 1024)`. No matching pretrained checkpoint is distributed here.

## Associated data and coordinates

Associated dataset destination: [HGG-VEM on Hugging Face](https://huggingface.co/datasets/yanchaoz/HGG-VEM). **The code PR does not upload the data or certify its current download availability.** Consult the dataset card and its file list for the actual deposited contents and access conditions.

The prepared data package contains nucleus-centred raw-intensity examples at **ZYX = 50 × 16 × 16 nm**, plus **3830 mitochondrial meshes**:

| Cell | Raw crop shape (Z × Y × X) | Mesh count |
| --- | --- | --- |
| 1 | 272 × 2300 × 2962 | 2676 |
| 2 | 170 × 732 × 1740 | 530 |
| 3 | 208 × 1074 × 944 | 624 |

These image examples come from an existing downsampled scale of the acquired 4-nm XY images. Crops cover approximately half the cell-body bounding-box XY area and half the original Z sections; this does not mean exactly half the labelled cell volume. Metadata records crop origins and boundary-driven centring adjustments.

Important coordinate rules:

- TIFF arrays use **ZYX**; CloudVolume uses **XYZ**.
- Legacy OBJ vertices store **(row Y, column X, Z), in nanometres**. Swap the first two coordinates for conventional XYZ. Do not multiply these vertices by voxel spacing a second time.
- Mesh coordinates refer to the full source volume, not a crop-local origin. The complete mesh cohort is not confined to the released image windows.
- Raw crop sampling (16-nm XY), source mesh-label sampling (8-nm XY), and contact computation sampling (4-nm XY) are distinct. Upsampling a public raw crop does not recreate the original contact-analysis inputs.

See [DATA_CONVENTIONS.md](docs/DATA_CONVENTIONS.md) for exact conversion formulas and table schemas.

## Validation and reproducibility

The release includes synthetic numerical tests and a recorded pre-release CPU report. Completed checks include contact-kernel agreement against independent nearest-neighbour calculations, anisotropic spacing and threshold inclusivity, synthetic instance reconstruction, SimSiam loss/gradient equivalence before and after renaming, and a PointNetV2 CPU forward pass.

Run the public tests:

```bash
python -m unittest discover -s tests -p 'test_numerical.py' -v
# Optional: requires torch plus the point-cloud utility dependencies.
python -m unittest discover -s tests -p 'test_pointcloud.py' -v
```

Tests use synthetic data and temporary outputs. They do not establish segmentation accuracy, biological replication, or clinical validity. The historical rename-parity check additionally used a private source snapshot; its recorded result is in `docs/TEST_REPORT.json`, not misrepresented as a fully public rerunnable comparison.

**Not verified by packaging:** a complete MIM runtime, GPU training, checkpoint-to-production replay, and reproduction of all manuscript metrics/figures. See [VALIDATION.md](docs/VALIDATION.md) for the test boundary. No training jobs run automatically when this repository is cloned.

## Acknowledgements and licensing

This work builds on [SparK](https://github.com/keyu-tian/SparK), [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1), [PyTorch Connectomics / MitoEM](https://github.com/PytorchConnectomics/pytorch_connectomics), PointNet++ implementations, and the [SimSiam learning formulation](https://github.com/facebookresearch/simsiam). The project-specific adaptation should not be confused with an official upstream implementation.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and bundled upstream licence texts. No new blanket licence for all HGG-specific code is assigned by this PR; upstream terms remain applicable. Please cite the underlying methods and record the HGG-VEM commit used in your analysis. A final manuscript citation/DOI should be added when available, rather than using a placeholder citation.

For questions or reproducible bug reports, use the repository issue tracker and include the module, commit, environment, array shapes, axis order, and a non-sensitive minimal example. Do not upload patient-identifying information, private file paths, or credentials.
