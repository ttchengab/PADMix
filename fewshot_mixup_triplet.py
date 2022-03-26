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

from v2vNet import Vox2Vox, VoxEncoder
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


rendering_dir = '/home/chengtim0708/shapenet/ShapeNetRendering/'
class V2VDataset(Dataset):
    def __init__(self, train=True):
        self.train = train
        if train == True:
            self.data = train_vox
        else:
            self.data = test_vox

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        curDir = rendering_dir + id_type_dir_convert[self.data[idx]['type']]['dir'] + '/' + self.data[idx]['dir'] + '/rendering/'
        img_file = random.choice([x for x in os.listdir(curDir) if '.png' in x])
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

        negative_voxel = []
        if self.train:
            nidx = random.randint(0, len(self.data)-1)
            mixupDir = rendering_dir + id_type_dir_convert[self.data[nidx]['type']]['dir'] + '/' + self.data[nidx]['dir'] + '/rendering/'
            mixupimg_file = random.choice([x for x in os.listdir(mixupDir) if '.png' in x])
            mixup_image_path = mixupDir + mixupimg_file
            mixup_image = cv2.imread(mixup_image_path)
            if np.max(mixup_image) > 1:
                mixup_image = mixup_image / 255
            mixup_image = torch.Tensor(mixup_image.transpose(2, 0, 1))
            mixup_image = torchvision.transforms.Resize(224)(mixup_image)
            mixup_voxel = voxel_id_mapping[id_type_dir_convert[self.data[nidx]['type']]['dir']+self.data[nidx]['dir']]
            mixup_voxel = mixup_voxel.unsqueeze(0)

            mixup_template_voxel =templateVoxels[self.data[nidx]['type']]
            mixup_template_voxel = mixup_template_voxel.unsqueeze(0)

            #Perform Mixup
            a = 0.4
            b = 0.4
            lam = np.random.beta(a, b)
            rendering_image = lam * rendering_image + (1 - lam) * mixup_image
            gt_voxel = lam * gt_voxel + (1 - lam) * mixup_voxel
            template_voxel = lam * template_voxel + (1 - lam) * mixup_template_voxel

            # Negative pair
            np_idx = random.randint(0, len(self.data)-1)
            while np_idx == idx or np_idx == nidx:
                np_idx = random.randint(0, len(self.data)-1)
            negative_voxel = voxel_id_mapping[id_type_dir_convert[self.data[np_idx]['type']]['dir']+self.data[np_idx]['dir']]
            negative_voxel = negative_voxel.unsqueeze(0)

        return rendering_image, template_voxel, gt_voxel, negative_voxel

BATCH_SIZE = 64
NUM_WORKERS = 16
V2V_LEARNING_RATE = 0.001
ENCODER_LEARNING_RATE = 0.0001
NUM_EPOCHES = 300

train_dataset = V2VDataset()
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

test_dataset = V2VDataset(train=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

#Focal ckpt mixup
checkpoint = torch.load('checkpoint_dir')


v2vnet = Vox2Vox()
gt_encoder = VoxEncoder()
gt_encoder.load_state_dict(checkpoint['encoder_state_dict'])
# v2vnet.load_state_dict(checkpoint['v2v_state_dict'])
v2vnet.decoder.load_state_dict(checkpoint['decoder_state_dict'])

if torch.cuda.is_available():
    v2vnet = torch.nn.DataParallel(v2vnet).cuda()
    gt_encoder = torch.nn.DataParallel(gt_encoder).cuda()

v2v_optimizer = torch.optim.Adam(v2vnet.parameters(), lr = V2V_LEARNING_RATE)
v2v_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(v2v_optimizer, mode='min', patience=5)

encoder_optimizer = torch.optim.Adam(gt_encoder.parameters(), lr = ENCODER_LEARNING_RATE)
encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', patience=5)

def FocalLoss(pred, target):
    eps = 1e-6
    alpha = 0.8
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    p_t = target * pred + (1 - target) * (1 - pred)
    gamma = 5
    return torch.mean(-alpha_t * torch.pow((1-p_t), gamma) * (target * torch.log(pred + eps) + (1 - target) * torch.log(1 - pred + eps)))

def cosSimLoss(pred, target):
    return torch.mean(1-torch.nn.CosineSimilarity()(pred, target))

def tripletLoss(pred, pos, neg):
    pos_sim = nn.CosineSimilarity()(pred, pos)
    neg_sim = nn.CosineSimilarity()(pred, neg)
    alpha = 0.2
    return torch.mean(torch.clamp((neg_sim - pos_sim + alpha), 0))


w_focal = 10
w_cossim = 0.5
w_triplet = 0.5
best_loss = 1000000
saving_dir = 'ckpts_fewshot/input0.4_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'/'
# model_dir = 'ckpts/MixupVis'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'/'
os.makedirs(saving_dir)
# os.makedirs(model_dir)


IOU_thres = [0.2, 0.25,0.3,0.35, 0.4, 0.5]

for epoch in range(NUM_EPOCHES):
    v2vnet.train()
    gt_encoder.train()
    f = open(saving_dir+'/IOUs.txt', 'a')
    f.write('Epoch '+str(epoch)+'\n')
    for idx, (images, template_voxels, gt_voxels, negative_voxels) in enumerate(tqdm(train_dataloader)):
        images = images.cuda()
        template_voxels = template_voxels.cuda()
        gt_voxels = gt_voxels.cuda()
        # gt_voxels128 = torch.ge(upsample128(gt_voxels), 0.5).float()
        negative_voxels = negative_voxels.cuda()

        encoded, out = v2vnet(images, template_voxels)
        gtEncoded = gt_encoder(gt_voxels).view(-1, 256*5*5*5)
        encoded = encoded.view(-1, 256*5*5*5)
        negEncoded = gt_encoder(negative_voxels).view(-1, 256*5*5*5)

        # focal_loss = FocalLoss(out, gt_voxels)
        bce_loss = nn.BCELoss()(out, gt_voxels)
        cos_loss = cosSimLoss(encoded, gtEncoded)
        triplet_loss = tripletLoss(encoded, gtEncoded, negEncoded)
        loss = w_focal * bce_loss + w_cossim * cos_loss + w_triplet * triplet_loss

        v2v_optimizer.zero_grad()
        encoder_optimizer.zero_grad()

        loss.backward()
        v2v_optimizer.step()
        encoder_optimizer.step()

        if idx % 10 == 0:
            print("Epoch {} Iteration {}, Current Loss: {}, bce: {}, tp: {}, cos: {}".format(epoch, idx, loss, bce_loss, triplet_loss, cos_loss))

    v2vnet.eval()
    gt_encoder.eval()

    checkpoint = {'v2v_state_dict': v2vnet.module.state_dict(),'encoder_state_dict': gt_encoder.module.state_dict()}
    torch.save(checkpoint, saving_dir+'/'+str(epoch)+'.pth')
    meanIOU = [0 for i in range(len(IOU_thres))]
    with torch.no_grad():
        total = 0
        val_loss = 0
        loss_total = 0
        iou_total = 0
        for idx, (images, template_voxels, gt_voxels, _) in enumerate(tqdm(test_dataloader)):
            images = images.cuda()
            template_voxels = template_voxels.cuda()
            gt_voxels = gt_voxels.cuda()
            encoded, out = v2vnet(images, template_voxels)
            gtEncoded = gt_encoder(gt_voxels).view(-1, 256*5*5*5)
            encoded = encoded.view(-1, 256*5*5*5)
            bce_loss = nn.BCELoss()(out, gt_voxels)
            cos_loss = cosSimLoss(encoded, gtEncoded)
            val_loss += w_focal * bce_loss + w_cossim * cos_loss

            for iouIdx, t in enumerate(IOU_thres):
                meanIOU[iouIdx] += computeIOU(out, gt_voxels, t, 32)

            loss_total += 1
            iou_total += len(images)
            
        for idx, t, in enumerate(IOU_thres):
            print("At Threshold {}, IOU: {}".format(t, meanIOU[idx]/iou_total))
            f.write("At Threshold {}, IOU (voxel): {}\n".format(t, meanIOU[idx]/iou_total))
        f.close()
        print("Validation loss is {}".format(val_loss/loss_total))
        if val_loss/total < best_loss:
            best_loss = val_loss/loss_total
            checkpoint = {'epoch': epoch, 'v2v_state_dict': v2vnet.module.state_dict(),'encoder_state_dict': gt_encoder.module.state_dict()()}
            torch.save(checkpoint, saving_dir+'/'+'best.pth')
        v2v_scheduler.step(val_loss)
        encoder_scheduler.step(val_loss)
