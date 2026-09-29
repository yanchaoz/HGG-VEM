from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import h5py
import logging
import argparse
import numpy as np
from attrdict import AttrDict

from tensorboardX import SummaryWriter
from collections import OrderedDict
import torch
import torch.nn as nn
from pretrain_provider_r import Provider
from utils.utils import setup_seed, execute
import torch.nn.functional as F
from visual2d import visual_2d
from nnUNet_head import Generic_UNet

def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            datefmt='%m-%d %H:%M',
            filename=path,
            filemode='w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    if not os.path.exists(cfg.cache_path):
        os.makedirs(cfg.cache_path)
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    if not os.path.exists(cfg.record_path):
        os.makedirs(cfg.record_path)
    if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer


def load_dataset(cfg):
    print('Caching datasets ... ', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider


def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters,
                                                                  cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(
                1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0

    from encoder3D import SparseEncoder
    from STUNet_head import STUNet
    from spark3D import SparK
    from decoder3D import LightDecoder

    head = Generic_UNet(
        input_channels=1,
        base_num_features=32,
        num_classes=5,
        num_pool=5,
        num_conv_per_stage=2,
        feat_map_mul_on_downscale=2,
        conv_op=nn.Conv3d,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-05, 'affine': True},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={'p': 0, 'inplace': True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'negative_slope': 0.01, 'inplace': True},
        deep_supervision=True,
        dropout_in_localization=False,
        final_nonlin=lambda x: x,  # 注意：你可以自定义 softmax_helper 或其他
        weightInitializer=None,
        pool_op_kernel_sizes=[[2, 2, 2]] * 5,
        conv_kernel_sizes=[[3, 3, 3]] * 6,
        upscale_logits=False,
        convolutional_pooling=True,
        convolutional_upsampling=True
    )


    input_size = (128, 128, 128)
    # print(head.state_dict().keys())
    enc = SparseEncoder(head, input_size=input_size, sbn=False)
    # print(enc.state_dict().keys())
    dec = LightDecoder(enc.downsample_ratio, sbn=False, width=512, out_channel=1)

    learner = SparK(
        sparse_encoder=enc, dense_decoder=dec, mask_ratio=0.75,
        densify_norm='in'
    )

    # print(learner.state_dict().keys())

    if cfg.MODEL.continue_train:
        ckpt_path = cfg.MODEL.continue_path
        print('Load pre-trained model from' + ckpt_path)
        checkpoint = torch.load(ckpt_path + '/learner.ckpt')
        new_state_dict = OrderedDict()
        state_dict = checkpoint['model_weights']
        for k, v in state_dict.items():
            name = k.replace('module.', '') if 'module' in k else k
            new_state_dict[name] = v
        learner.load_state_dict(new_state_dict)

    learner = learner.cuda()

    optimizer = torch.optim.AdamW(learner.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999))


    while iters <= cfg.TRAIN.total_iters:
        learner.train()
        iters += 1
        t1 = time.time()
        inputs = train_provider.next()
        inputs = inputs.cuda()
        optimizer.zero_grad()

        rec, inp, non_active = learner(inputs)
        l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)
        loss = l2_loss.mul_(non_active).sum() / (non_active.sum() + 1e-8)  # loss only on masked (non-active) patches

        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        sum_time += time.time() - t1

        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info(
                    'step %d, loss = %.6f (wt: *1, et: %.2f sec, rd: %.2f min)'
                    % (iters, sum_loss, sum_time,
                       (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss * 1, iters)
            else:
                logging.info(
                    'step %d, loss = %.6f (wt: *1, et: %.2f sec, rd: %.2f min)' \
                    % (iters, sum_loss / cfg.TRAIN.display_freq * 1, sum_time, \
                       (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)
            f_loss_txt.write('step = %d, loss = %.6f'% (iters, sum_loss / cfg.TRAIN.display_freq * 1))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0

        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            visual_2d(learner, inputs, cfg.cache_path, iters)
        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                      'model_weights': learner.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'learner.ckpt'))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)

    f_loss_txt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='Base_MAE', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider = load_dataset(cfg)
        init_iters = 0
        loop(cfg, train_provider, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')
