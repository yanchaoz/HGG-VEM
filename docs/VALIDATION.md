# Validation scope

Validation is split into numerical self-tests, public synthetic interface tests, and pre-release checks recorded against the supplied research-source snapshot. None of these is a replacement for biological validation or a rerun of segmentation cross-validation.

## Public, rerunnable tests

From the repository root:

```bash
python -m unittest discover -s tests -p 'test_numerical.py' -v
python -m unittest discover -s tests -p 'test_pointcloud.py' -v
```

| Test group | What it checks | What it does not establish |
| --- | --- | --- |
| Source syntax | All module Python files parse | Import-time compatibility of every training dependency |
| Threshold kernel | Independent KD-tree agreement, empty targets, Z anisotropy, inclusive thresholds, finite halos, integer mapping | Correct acquisition registration for a new dataset |
| Full-range kernel | 48 independent explicitly expanded-grid comparisons | Availability of production masks |
| Instance synthetic tests | Two-object separation; boundary expansion changes seeds without eroding support | Production split/merge rates |
| Instance CLI | NumPy/TIFF input, output shape/type, settings file, overwrite refusal, invalid input rejection | Full network inference or distributed stitching |
| Primary-table adapter | Complete partner pairs, fixed-count gate, censoring, threshold consistency, metadata preservation | A new morphology classification |
| PointNet++ forward | Finite `(2,1024)` representation from `(2,1024,3)` input | A trained or accurate encoder |
| SimSiam backward | Finite objective and parameter gradients on a tiny CPU encoder | Convergence or reproduced downstream performance |
| Mesh utility | Synthetic OBJ → normalized HDF5, source filename, finite samples, unit radius | Identity with historical training point clouds |

Optional point-cloud tests report explicit skips if their dependencies are absent. A skipped test is not a pass. Tests create only temporary synthetic data; GPU training and clinical data are not used.

## Recorded research-source checks

`TEST_REPORT.json` records CPU checks completed before this PR. The renamed SimSiam wrapper was compared with the supplied original wrapper configured with `use_momentum=False`, using identical initialization and inputs. Recorded losses were both `3.948773145675659`; the maximum parameter-gradient difference was `0.0`. This comparison isolated the wrapper with a tiny encoder; a separate test checked the real PointNetV2 forward shape. The private original-source snapshot is not included, so this recorded rename comparison is not presented as a public rerunnable test.

The PR also records a fresh run of the public tests in `PR_TEST_REPORT.json`, including versions and skip/failure counts. It should be read separately from the historical packaging report; neither report certifies unsupported training environments.

## Not verified by this release process

- Full MIM import/runtime execution with the historical nnU-Net/timm environment.
- MIM or point-cloud GPU pretraining, convergence, checkpoint-to-production replay, or complete resume behaviour.
- Reproduction of manuscript segmentation scores, morphology measurements, classifications, and all figures.
- Patient-independent generalization, functional MCS identity, or clinical utility.
- Current Hugging Face download availability or data-sharing approval.

The released contact kernels preserve study-specific conventions. Do not change cohort IDs, voxel spacing, image origins, dilation, threshold inclusivity, or censoring silently. If adapting the method, record the changes and rerun appropriate input and numerical checks.
