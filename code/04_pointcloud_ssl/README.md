# HGG mitochondrial SimSiam

Source: the author-selected `unified_ssl/train` implementation. Entry point: `python train_simsiam.py -c hgg_simsiam` from this directory.

The encoder is PointNet++ / PointNetV2 and emits 1024-dimensional features. `simsiam.py::SimSiam` uses a 128-dimensional projection and a 512-wide hidden layer, with a shared stop-gradient target branch and **no EMA target encoder**. The original source class was named BYOL but configured with `use_momentum=False`; the release name reflects that actual algorithm. Original two-direction normalized squared-distance loss and its scale are retained.

Input is an HDF5 file with dataset `cloudpoints`, shape `(objects, points, 3)`. Set `DATA.pointcloud_h5` in the config. Two stochastic views are independently augmented, farthest-point sampled to 1024 points and normalized. The original loader excluded the last 16 objects; `DATA.holdout_last_n` explicitly preserves that default. Do not assume that it defines a patient-independent validation split.

`mesh_to_pointcloud.py` is a **new release utility** for preparing HDF5 inputs from the public OBJ meshes, not an asserted reconstruction of the historical training corpus. It writes the cell/instance IDs alongside point clouds; mesh coordinates are preserved until centring/normalization for training. Run with `--help` for arguments.

`train_simsiam.py` saves encoder checkpoints during training. `extract_embeddings.py` loads a matching encoder checkpoint; configure the input and model paths before use. Legacy resume logic does not constitute full optimizer-state restoration; leave `TRAIN.resume: false` for this candidate. No pretrained checkpoint from another architecture is silently paired with this encoder. No pretraining run or performance reproduction was conducted during packaging.
