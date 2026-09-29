# Third-party acknowledgements and licence notices

HGG-VEM contains adaptations assembled from the research code supplied by the authors. Project-specific naming does not imply that the underlying architectures or upstream utilities originated in this repository. Preserve the following notices when redistributing the relevant components.

| Component | Upstream reference | Included licence text |
| --- | --- | --- |
| Sparse masked-modeling implementation | [SparK](https://github.com/keyu-tian/SparK) | [MIT; Keyu Tian](licenses/SparK.txt) |
| U-Net architecture / nnU-Net v1 utilities | [nnU-Net v1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) | [Apache-2.0; DKFZ](licenses/nnUNet-v1.txt) |
| Retained STU-Net auxiliary definitions | [STU-Net](https://github.com/uni-medical/STU-Net) | [Apache-2.0](licenses/STU-Net.txt) |
| Instance post-processing kernels | [PyTorch Connectomics](https://github.com/PytorchConnectomics/pytorch_connectomics) and the MitoEM attribution retained in `process.py` | [MIT; PyTorch Connectomics Contributors](licenses/PyTorch-Connectomics.txt) |
| PointNet++ Python utility implementation | [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) | [MIT; benny](licenses/PointNet2.txt) |
| BYOL-style wrapper utilities retained in the non-momentum adaptation | [byol-pytorch](https://github.com/lucidrains/byol-pytorch) | [MIT; Phil Wang](licenses/BYOL-pytorch.txt) |

The active point-cloud method is **SimSiam**, with a shared stop-gradient encoder and no momentum target. The [official SimSiam project](https://github.com/facebookresearch/simsiam) is a method reference, not a claim that the local training entry is an unchanged copy of that repository.

The U-Net adaptation selects the study's encoder outputs and integrates with the 3D masked-modeling pipeline. Packaging removes host-specific import paths and exposes local configuration. The SimSiam adaptation removes the unused momentum path, retains the active research objective, and renames the entry points. Instance and proximity methods are described in their module READMEs. See [PACKAGING_CHANGES.md](docs/PACKAGING_CHANGES.md).

`licenses/SOURCES.json` records the public locations and Git blob hashes from which the licence texts were retrieved. It is **not** a claim that the locally supplied research files correspond to those repositories' current HEAD revisions. Original in-file citations, including the MitoEM attribution and the EM augmentation / bounding-box utility references, are preserved.

Dependencies installed through `requirements.txt` retain their own licences. The included third-party texts do not assign a blanket licence to every HGG-specific contribution, the image data, the mesh data, or unpublished model weights. The maintainers should complete their project-level licence and source-provenance review before treating this PR as a fully licensed release. No blanket relicensing is performed here.
