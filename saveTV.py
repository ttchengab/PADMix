import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import tqdm
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

# from v2vNet import VoxEncoder, VoxDecoder
import utils.binvox_rw
from helper import plot_pointcloud, cubifyAndPlot
avgpool = nn.AdaptiveAvgPool3d((128, 128, 128))

with open('dataset/ShapeNet.json') as json_file:
    data = json.load(json_file)

template_voxels = []
voxel_dir = '/home/chengtim0708/shapenet/ShapeNetVox32/'
novel_classes = [1, 2, 6, 8, 9, 12]
for idx, i in enumerate(data):
    curTemplate = []
    for objdir in i['train']:
        voxpath = voxel_dir + i['taxonomy_id'] +  '/' + objdir + '/model.binvox'
        if os.path.exists(voxpath):
            with open(voxpath, 'rb') as f:
                volume = utils.binvox_rw.read_as_3d_array(f)
                volume = torch.Tensor(volume.data.astype(np.float32))
            curTemplate.append(volume)
            if idx in novel_classes:
                break
    template_voxels.append(torch.ge(torch.mean(torch.stack(curTemplate), dim=0), 0.2).float())
template_names = ['aeroplane', 'bench', 'cabinet', 'car', 'chair', 'display', 'lamp', 'speaker', 'rifle', 'sofa', 'table', 'telephone', 'watercraft' ]

datasets_dir = 'dataset/'
with open('dataset/ShapeNet.json') as json_file:
    taxonomies = json.load(json_file)
id_type_dir_convert = []
for i in taxonomies:
    id_type_dir_convert.append({'type': i['taxonomy_name'], 'dir': i['taxonomy_id']})

with open(datasets_dir+'1shot_base_final.txt') as json_file:
    fewshot_vox = json.load(json_file)

for entry in fewshot_vox:
    voxpath = voxel_dir + id_type_dir_convert[entry['type']]['dir'] + '/' + entry['dir'] + '/model.binvox'
    if os.path.exists(voxpath):
        with open(voxpath, 'rb') as f:
            volume = utils.binvox_rw.read_as_3d_array(f)
            volume = torch.Tensor(volume.data.astype(np.float32))
        template_voxels[entry['type']] = volume

for idx, curTemplate in enumerate(template_voxels):
    torch.save(curTemplate, 'template/' + template_names[idx] + '.pt')
# for idx, curTemplate in enumerate(template_voxels):
#     cubifyAndPlot(curTemplate.unsqueeze(0), str(idx))
