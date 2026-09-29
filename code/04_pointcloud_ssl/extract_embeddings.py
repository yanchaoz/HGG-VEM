import os
import cv2
import h5py
import yaml
import torch
import argparse
import numpy as np
from skimage import morphology
from attrdict import AttrDict
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from data_utils.pcloader import ModelNetDataLoader
from model.pointnetv2_encoder import PointNetV2
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='hgg_simsiam', help='config basename under config/')
    parser.add_argument('-mn', '--model_name', type=str, default='hgg_simsiam')
    parser.add_argument('-id', '--model_id', type=str, default='model-080000')
    parser.add_argument('-m', '--mode', type=str, default='')
    parser.add_argument('-s', '--save', action='store_false', default=True)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name
    out_path = os.path.join('./inference', trained_model, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'feat_' + args.model_id
    out_affs = os.path.join(out_path, img_folder)
    if not os.path.exists(out_affs):
        os.makedirs(out_affs)
    print('out_path: ' + out_affs)

    device = torch.device('cuda:0')
    print('load PointNetV2!')
    model = PointNetV2().to(device)

    ckpt_path = os.path.join('./trained_model', trained_model, args.model_id + '.ckpt')
    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        name = k.replace('module.', '') if 'module' in k else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)

    valid_provider = ModelNetDataLoader(cfg.DATA.pointcloud_h5, npoint=1024, uniform=False, istrain=False,
                                       holdout_last_n=cfg.DATA.holdout_last_n)
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1)
    model.eval()

    print('the number of points:', len(valid_provider))
    losses_valid = []
    pbar = tqdm(total=len(valid_provider))

    feature_list = []
    for kk, data in enumerate(val_loader, 0):
        inputs = data
        inputs = inputs.cuda()
        with torch.no_grad():
            representation = model(inputs)
            feature_list.append(representation.cpu().numpy())
        pbar.update(1)
    pbar.close()

    np.save(os.path.join(out_affs, 'features.npy'), feature_list)
