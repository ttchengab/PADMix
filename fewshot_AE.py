import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision

import random
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from tqdm import tqdm
import skimage.measure
import scipy
import scipy.io
from scipy import ndimage
import math
import json
import numpy as np
import os
import glob
from datetime import datetime

from v2vNet import VoxEncoder, VoxDecoder
import utils.binvox_rw
from helper import plot_pointcloud, cubifyAndPlot, computeIOU

# upsample128 = nn.AdaptiveAvgPool3d((128, 128, 128))
# downsample32 = nn.AdaptiveMaxPool3d((32, 32, 32))

# Load all taxonomies and their conversion
with open('dataset/ShapeNet.json') as json_file:
    taxonomies = json.load(json_file)

id_type_dir_convert = []
for i in taxonomies:
    id_type_dir_convert.append({'type': i['taxonomy_name'], 'dir': i['taxonomy_id']})

print("Loading Template Voxels")
templates_dir = 'template/'
templateVoxels = []
for curType in id_type_dir_convert:
    templateVoxels.append(torch.load(templates_dir+curType['type']+'.pt'))

# print("Loading Training and Testing Data")
datasets_dir = 'dataset/'

print("Loading All Voxels")
with open(datasets_dir+'fewshot_base.txt') as json_file:
    train_vox = json.load(json_file)

with open(datasets_dir+'1shot_base_final.txt') as json_file:
    fewshot_vox = json.load(json_file)
train_vox.extend(fewshot_vox)

with open(datasets_dir+'few_shot_test.txt') as json_file:
    test_vox = json.load(json_file)

voxel_dir = '/home/chengtim0708/shapenet/ShapeNetVox32/'
voxel_id_mapping = {}
print("Loading Training Voxels")
for entry in tqdm(train_vox):
    gt_voxel_path = voxel_dir + id_type_dir_convert[entry['type']]['dir'] + '/' + entry['dir'] + '/model.binvox'
    with open(gt_voxel_path, 'rb') as f:
        gt_voxel = utils.binvox_rw.read_as_3d_array(f)
        gt_voxel = torch.Tensor(gt_voxel.data.astype(np.float32))
    voxel_id_mapping[id_type_dir_convert[entry['type']]['dir']+entry['dir']] = gt_voxel

print("Loading Testing Voxels")
for entry in tqdm(test_vox):
    gt_voxel_path = voxel_dir + id_type_dir_convert[entry['type']]['dir'] + '/' + entry['dir'] + '/model.binvox'
    with open(gt_voxel_path, 'rb') as f:
        gt_voxel = utils.binvox_rw.read_as_3d_array(f)
        gt_voxel = torch.Tensor(gt_voxel.data.astype(np.float32))
    voxel_id_mapping[id_type_dir_convert[entry['type']]['dir']+entry['dir']] = gt_voxel


class V2VDataset(Dataset):
    def __init__(self, train=True):
        if train == True:
            self.data = train_vox
        else:
            self.data = test_vox

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        gt_voxel = voxel_id_mapping[id_type_dir_convert[self.data[idx]['type']]['dir']+self.data[idx]['dir']]
        gt_voxel = gt_voxel.unsqueeze(0)
        # gt_voxel128 = torch.ge(upsample128(gt_voxel), 0.5).float()
        return gt_voxel

BATCH_SIZE = 64
NUM_WORKERS = 16
V2V_LEARNING_RATE = 0.001
NUM_EPOCHES = 300

train_dataset = V2VDataset()
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

test_dataset = V2VDataset(train=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

gt_encoder = VoxEncoder()
gt_decoder = VoxDecoder()

if torch.cuda.is_available():
    # v2vnet = torch.nn.DataParallel(v2vnet).cuda()
    gt_encoder = torch.nn.DataParallel(gt_encoder).cuda()
    gt_decoder = torch.nn.DataParallel(gt_decoder).cuda()

gt_optimizer = torch.optim.Adam(gt_encoder.parameters(), lr = V2V_LEARNING_RATE)
gt_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gt_optimizer, mode='min', patience=5)

gtd_optimizer = torch.optim.Adam(gt_decoder.parameters(), lr = V2V_LEARNING_RATE)
gtd_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gtd_optimizer, mode='min', patience=5)


def FocalLoss(pred, target):
    eps = 1e-6
    alpha = 0.8
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    p_t = target * pred + (1 - target) * (1 - pred)
    gamma = 5
    return torch.mean(-alpha_t * torch.pow((1-p_t), gamma) * (target * torch.log(pred + eps) + (1 - target) * torch.log(1 - pred + eps)))

def cosSimLoss(pred, target):
    return torch.mean(1-torch.nn.CosineSimilarity()(pred, target))

w_focal = 10
w_cossim = 0.5
best_loss = 1000000
saving_dir = 'ckpts/1_Shot_Autoencoder'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'/'
# model_dir = 'ckpts/MixupVis'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'/'
os.makedirs(saving_dir)
# os.makedirs(model_dir)


IOU_thres = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]

for epoch in range(NUM_EPOCHES):
    gt_encoder.train()
    gt_decoder.train()
    f = open(saving_dir+'/IOUs.txt', 'a')
    f.write('Epoch '+str(epoch)+'\n')
    for idx, gt_voxels in enumerate(tqdm(train_dataloader)):
        gt_voxels = gt_voxels.cuda()
        # gt_voxels128 = torch.ge(upsample128(gt_voxels), 0.5).float()

        gtEncoded = gt_encoder(gt_voxels)
        out = gt_decoder(gtEncoded)

        focal_loss = FocalLoss(out, gt_voxels)
        # focal_loss = nn.BCELoss()(out, gt_voxels)
        loss = w_focal * focal_loss

        gt_optimizer.zero_grad()
        gtd_optimizer.zero_grad()
        # encoder_optimizer.zero_grad()

        loss.backward()
        gt_optimizer.step()
        gtd_optimizer.step()

        if idx % 10 == 0:
            print("Epoch {} Iteration {}, Current Loss: {}, focalLoss: {}".format(epoch, idx, loss, focal_loss))

    gt_encoder.eval()
    gt_decoder.eval()
    checkpoint = {'decoder_state_dict': gt_decoder.module.state_dict(),'encoder_state_dict': gt_encoder.module.state_dict()}
    torch.save(checkpoint, saving_dir+'/'+str(epoch)+'.pth')
    meanIOU = [0 for i in range(len(IOU_thres))]
    with torch.no_grad():
        total = 0
        val_loss = 0
        loss_total = 0
        iou_total = 0
        for idx, (gt_voxels) in enumerate(tqdm(test_dataloader)):
            gt_voxels = gt_voxels.cuda()
            gtEncoded = gt_encoder(gt_voxels)

            out = gt_decoder(gtEncoded)
            focal_loss = FocalLoss(out, gt_voxels)
            val_loss += w_focal * focal_loss

            for iouIdx, t in enumerate(IOU_thres):
                meanIOU[iouIdx] += computeIOU(out, gt_voxels, t, 32)

            loss_total += 1
            iou_total += len(gt_voxels)
        for idx, t, in enumerate(IOU_thres):
            print("At Threshold {}, IOU: {}".format(t, meanIOU[idx]/iou_total))
            f.write("At Threshold {}, IOU (voxel): {}\n".format(t, meanIOU[idx]/iou_total))
        f.close()
        print("Validation loss is {}".format(val_loss/loss_total))
        if val_loss/total < best_loss:
            best_loss = val_loss/loss_total
            checkpoint = {'epoch': epoch, 'decoder_state_dict': gt_decoder.module.state_dict(),'encoder_state_dict': gt_encoder.module.state_dict()()}
            torch.save(checkpoint, saving_dir+'/'+'best.pth')
        gt_scheduler.step(val_loss)
        gtd_scheduler.step(val_loss)
