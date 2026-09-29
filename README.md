# HGG-VEM

Computational methods for volume electron microscopy of high-grade glioma.

## Modules

| Module | Purpose | Documentation |
| --- | --- | --- |
| Masked image modeling | SparK-style pretraining with a 3D U-Net encoder | [Guide](code/01_masked_image_modeling/README.md) |
| Mitochondrial instances | Foreground and boundary probabilities to instance labels | [Guide](code/02_mitochondrial_instances/README.md) |
| Organelle proximity | Mitochondria–ER and mitochondria–Golgi distances from input labels | [Guide](code/03_contact_analysis/README.md) |
| Point-cloud learning | PointNet++ representations trained with SimSiam | [Guide](code/04_pointcloud_ssl/README.md) |

```text
code/
├── 01_masked_image_modeling/
├── 02_mitochondrial_instances/
├── 03_contact_analysis/
└── 04_pointcloud_ssl/
docs/       # Input conventions and validation details
tests/      # Synthetic numerical and interface tests
```

## Installation

```bash
git clone https://github.com/yanchaoz/HGG-VEM.git
cd HGG-VEM
python -m venv .venv
source .venv/bin/activate
```

Install dependencies for the module you need, for example:

```bash
python -m pip install -r code/03_contact_analysis/requirements.txt
```

Use separate environments for numerical analysis and training. The training modules use a Python 3.8-era stack; MIM requires nnU-Net **v1** and `timm==0.6.13`. Install a PyTorch/torchvision pair compatible with your CUDA environment. Module-specific requirements and compatibility notes are provided in each guide.

## Usage

Run the following examples from the corresponding module directory. Configure input paths before launching training or full-volume analysis.

### Masked image modeling

```bash
cd code/01_masked_image_modeling
python pretrain_nn.py -c Base_MAE
```

Set `DATA.data_folder` in `config/Base_MAE.yaml` to a directory of TIFF stacks. The default model uses 128³ grayscale patches, a 0.75 mask ratio, and masked reconstruction error. Adjust worker counts to the available memory.

### Mitochondrial instance reconstruction

```bash
cd code/02_mitochondrial_instances
python instances_from_probabilities.py \
  --foreground /path/to/foreground.tif \
  --boundary /path/to/boundary.tif \
  --output /path/to/new_instances.tif
```

Inputs are aligned ZYX probability volumes; outputs are integer instance labels and a settings sidecar. Reconstruction options and probability conventions are described in the module guide.

### Organelle proximity

```bash
cd code/03_contact_analysis
python recompute_mask_proximity.py --self-test
python extend_fullrange_proximity.py --self-test
```

Distances are measured directly between supplied mitochondrial and partner labels on a **50 × 4 × 4 nm (ZYX)** computation grid. Threshold calls use **≤ 30, 50, 80, and 100 nm**. These calls describe morphological proximity, not functional confirmation of membrane contact sites. See the [full-volume workflow](code/03_contact_analysis/README.md#full-volume-workflow) for registered-mask inputs, cohort tables, and output formats.

### Point-cloud SimSiam

```bash
cd code/04_pointcloud_ssl
python mesh_to_pointcloud.py \
  --mesh-root /path/to/extracted_meshes \
  --output ./data/mitochondrial_pointclouds.h5
python train_simsiam.py -c hgg_simsiam
```

Set `DATA.pointcloud_h5` in `config/hgg_simsiam.yaml`. The PointNet++ encoder produces 1024-dimensional features; SimSiam uses two augmented views and a stop-gradient target. Checkpoint layout, embedding extraction, and input-order exclusions are documented in the module guide.

## Data and coordinates

Associated data project: [HGG-VEM on Hugging Face](https://huggingface.co/datasets/yanchaoz/HGG-VEM). Refer to its dataset card and file list for deposited contents and access conditions.

The prepared package contains representative EM crops at **50 × 16 × 16 nm (ZYX)** and **2676 / 530 / 624 mitochondrial meshes** for Cells 1–3. Image and mesh files are managed separately from this code repository.

- TIFF arrays use **ZYX**; CloudVolume indexing uses **XYZ**.
- Legacy OBJ vertices store **(Y, X, Z) in nanometres**. Swap the first two coordinates for XYZ; do not multiply by voxel spacing again.
- Crop origins refer to the full source volume. Meshes are not restricted to the image windows.
- Image-release sampling, source-label sampling, and the distance-computation grid are distinct.

See [data conventions](docs/DATA_CONVENTIONS.md) for coordinate conversion and input schemas.

## Tests

```bash
python -m pip install -r code/02_mitochondrial_instances/requirements.txt
python -m unittest discover -s tests -p 'test_numerical.py' -v
# Optional: install the point-cloud dependencies first.
python -m unittest discover -s tests -p 'test_pointcloud.py' -v
```

Tests use synthetic inputs. Environment details, completed checks, and untested workflows are listed in [validation](docs/VALIDATION.md). Pretrained weights, complete training data, and the formal morphometric measurement pipeline are not included.

## Acknowledgements

The implementation builds on SparK, nnU-Net, PyTorch Connectomics, PointNet++, and SimSiam. See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream references and applicable licences.
