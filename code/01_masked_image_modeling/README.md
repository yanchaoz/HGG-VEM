# Masked image modeling (3D U-Net default)

Entry point: `python pretrain_nn.py -c Base_MAE` from this directory.

The supplied source builds `Generic_UNet` from `nnUNet_head.py`, wraps its encoder in `SparseEncoder`, and reconstructs masked patches through `LightDecoder` and `SparK`. Default input is 128 x 128 x 128 voxels, one grayscale channel; the training entry uses a mask ratio of 0.75 and masked-patch squared reconstruction error. The default is **not** STU-Net or MedNeXt; STUNet_head is retained only for existing ancillary imports.

Set `DATA.data_folder` in `config/Base_MAE.yaml` to a directory containing subdirectories of 3D image stacks. The supplied loader discards 12 pixels from each XY edge and then samples 128-cubed crops, so each input stack must have Z >= 128 and Y/X >= 152. The three nucleus-centred release crops satisfy the shape requirement but are illustrative data, not the original pretraining corpus. The loader ignores metadata and preview files. Inputs are uint8 grayscale and divided by 255.

GPU training is the historical default. No training is launched during packaging. Dependency versions below are a starting environment specification, not a claim of a fully reproduced historical environment.

Run `pip install -r requirements.txt` in a suitable environment. The `nnunet` dependency refers to **nnU-Net v1**. All in-repository imports assume this module directory is the working directory. `config/Base_MAE.yaml` contains portable paths; set `CUDA_VISIBLE_DEVICES` externally as needed.
