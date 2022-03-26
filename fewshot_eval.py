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

from v2vNet import Vox2Vox, VoxEncoder, Img2Vox
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
print("Loading Testing Voxels")
for entry in tqdm(test_vox):
    gt_voxel_path = voxel_dir + id_type_dir_convert[entry['type']]['dir'] + '/' + entry['dir'] + '/model.binvox'
    with open(gt_voxel_path, 'rb') as f:
        gt_voxel = utils.binvox_rw.read_as_3d_array(f)
        gt_voxel = torch.Tensor(gt_voxel.data.astype(np.float32))
    voxel_id_mapping[id_type_dir_convert[entry['type']]['dir']+entry['dir']] = gt_voxel


rendering_dir = '/home/chengtim0708/shapenet/ShapeNetRendering/'
class V2VDataset(Dataset):
    def __init__(self, type):
        self.data = [entry for entry in test_vox if entry['type'] == type]

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        curDir = rendering_dir + id_type_dir_convert[self.data[idx]['type']]['dir'] + '/' + self.data[idx]['dir'] + '/rendering/'
        # img_file = random.choice([x for x in os.listdir(curDir) if '.png' in x])
        img_file = [x for x in os.listdir(curDir) if '.png' in x][0]
        rendering_image_path = curDir + img_file

        rendering_image = cv2.imread(rendering_image_path)
        if np.max(rendering_image) > 1:
            rendering_image = rendering_image / 255
        rendering_image = torch.Tensor(rendering_image.transpose(2, 0, 1))
        rendering_image = torchvision.transforms.Resize(224)(rendering_image)

        template_voxel = templateVoxels[self.data[idx]['type']]
        template_voxel = template_voxel.unsqueeze(0)

        gt_voxel = voxel_id_mapping[id_type_dir_convert[self.data[idx]['type']]['dir']+self.data[idx]['dir']]
        gt_voxel = gt_voxel.unsqueeze(0)

        return rendering_image, template_voxel, gt_voxel

BATCH_SIZE = 64
NUM_WORKERS = 16

checkpoint = torch.load('put_checkpoint_here')
v2vnet = Vox2Vox()
# v2vnet = Img2Vox()
gt_encoder = VoxEncoder()
#
gt_encoder.load_state_dict(checkpoint['encoder_state_dict'])
v2vnet.load_state_dict(checkpoint['v2v_state_dict'])

if torch.cuda.is_available():
    v2vnet.cuda()
    gt_encoder.cuda()
    # v2vnet = torch.nn.DataParallel(v2vnet).cuda()
    # gt_encoder = torch.nn.DataParallel(gt_encoder).cuda()


v2vnet.eval()
gt_encoder.eval()

IOU_thres = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
novel_base = [1,2,6,8,9,12]

for category in novel_base:
    test_dataset = V2VDataset(type=category)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print('For Type {}'.format(id_type_dir_convert[category]['type']))
    meanIOU = [0 for i in range(len(IOU_thres))]
    with torch.no_grad():
        total = 0
        iou_total = 0
        for idx, (images, template_voxels, gt_voxels) in enumerate(tqdm(test_dataloader)):
            images = images.cuda()
            template_voxels = template_voxels.cuda()
            gt_voxels = gt_voxels.cuda()
            encoded, out = v2vnet(images, template_voxels)
            # encoded, out = v2vnet(images)
            gtEncoded = gt_encoder(gt_voxels).view(-1, 256*5*5*5)
            encoded = encoded.view(-1, 256*5*5*5)

            for iouIdx, t in enumerate(IOU_thres):
                meanIOU[iouIdx] += computeIOU(out, gt_voxels, t, 32)
            iou_total += len(images)

        for idx, t, in enumerate(IOU_thres):
            print("At Threshold {}, IOU: {}".format(t, meanIOU[idx]/iou_total))
