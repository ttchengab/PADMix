import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm, ReLU
import torchvision
from torch.distributions.beta import Beta
class VoxEncoder(nn.Module):
    def __init__(self):
        super(VoxEncoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=4)
        self.batchnorm1 = nn.BatchNorm3d(32)
        self.batchnorm2 = nn.BatchNorm3d(64)
        self.batchnorm3 = nn.BatchNorm3d(128)
        self.batchnorm4 = nn.BatchNorm3d(256)
        self.activation = nn.ReLU()
        self.avgpool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.activation(self.batchnorm1(self.conv1(x)))
        # x = self.avgpool(x)
        x = self.activation(self.batchnorm2(self.conv2(x)))
        # x = self.avgpool(x)
        x = self.activation(self.batchnorm3(self.conv3(x)))
        # x = self.avgpool(x)
        x = self.activation(self.batchnorm4(self.conv4(x)))
        # print(x.shape)
        return x


class VoxDecoder(nn.Module):
    def __init__(self):
        super(VoxDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2)
        self.conv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2)
        self.conv4 = nn.ConvTranspose3d(32, 1, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm3d(128)
        self.batchnorm2 = nn.BatchNorm3d(64)
        self.batchnorm3 = nn.BatchNorm3d(32)
        self.ReLU = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool3d((62, 62, 62))
    def forward(self, x):
        x = self.ReLU(self.batchnorm1(self.conv1(x)))
        x = self.ReLU(self.batchnorm2(self.conv2(x)))
        x = self.avgpool(x)
        x = self.ReLU(self.batchnorm3(self.conv3(x)))
        x = self.conv4(x)
        return nn.Sigmoid()(x)
        # return nn.Sigmoid()(x+template)

class merger(nn.Module):
    def __init__(self):
        super(merger, self).__init__()
        self.mergeConv1 = nn.Conv3d(280, 272, kernel_size=3, padding=1)
        self.mergeConv2 = nn.Conv3d(272, 264, kernel_size=3, padding=1)
        self.mergeConv3 = nn.Conv3d(264, 256, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(272)
        self.batchnorm2 = nn.BatchNorm3d(264)
        self.batchnorm3 = nn.BatchNorm3d(256)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.activation(self.batchnorm1(self.mergeConv1(x)))
        x = self.activation(self.batchnorm2(self.mergeConv2(x)))
        x = self.activation(self.mergeConv3(x))
        return x


class mergerImage(nn.Module):
    def __init__(self):
        super(mergerImage, self).__init__()
        self.mergeConv1 = nn.Conv3d(24, 256, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.activation(self.mergeConv1(x))
        return x

class Vox2VoxImage(nn.Module):
    def __init__(self):
        super(Vox2VoxImage, self).__init__()
        resnet = torchvision.models.resnet34(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:6])
        self.avgpool = nn.AdaptiveAvgPool2d((18,18))
        self.merger = mergerImage()
        self.decoder = VoxDecoder()

        self.ReLU = nn.ReLU()


    def forward(self, imgs, voxels):
        # Nx3x224x224 --> Nx128x28x28
        features = self.resnet(imgs)
        features = self.avgpool(features)
        features = features.view(-1, 24, 12, 12, 12)
        # print(features)
        # Nx1x128x128x128 --> Nx8x12x12x12

        # Nx32x12x12x12 --> Nx8x12x12x12
        combinedFeatures = self.merger(features)

        out = self.decoder(combinedFeatures, voxels)

        return combinedFeatures, out

class Vox2Vox(nn.Module):
    def __init__(self):
        super(Vox2Vox, self).__init__()
        resnet = torchvision.models.resnet34(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:6])
        self.avgpool = nn.AdaptiveAvgPool2d((18,18))

        self.encoder = VoxEncoder()
        self.merger = merger()
        self.decoder = VoxDecoder()

        self.ReLU = nn.ReLU()


    def forward(self, imgs, voxels):
        # Nx3x224x224 --> Nx128x28x28
        features = self.resnet(imgs)
        features = self.avgpool(features)
        features = features.view(-1, 24, 12, 12, 12)
        # print(features)
        # Nx1x128x128x128 --> Nx8x12x12x12
        voxelFeatures = self.encoder(voxels)

        # Nx32x12x12x12 --> Nx8x12x12x12
        combinedFeatures = torch.cat([voxelFeatures, features], dim=1)
        combinedFeatures = self.merger(combinedFeatures)

        out = self.decoder(combinedFeatures)

        return combinedFeatures, out



class mergerK(nn.Module):
    def __init__(self):
        super(mergerK, self).__init__()
        self.mergeConv1 = nn.Conv3d(281, 272, kernel_size=3, padding=1)
        self.mergeConv2 = nn.Conv3d(272, 264, kernel_size=3, padding=1)
        self.mergeConv3 = nn.Conv3d(264, 256, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(272)
        self.batchnorm2 = nn.BatchNorm3d(264)
        self.batchnorm3 = nn.BatchNorm3d(256)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.activation(self.batchnorm1(self.mergeConv1(x)))
        x = self.activation(self.batchnorm2(self.mergeConv2(x)))
        x = self.activation(self.mergeConv3(x))
        return x

class Vox2Vox_Mixup(nn.Module):
    def __init__(self):
        super(Vox2Vox_Mixup, self).__init__()
        resnet = torchvision.models.resnet34(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:6])
        self.avgpool = nn.AdaptiveAvgPool2d((18,18))
        self.encoder = VoxEncoder()
        self.merger = merger()
        self.decoder = VoxDecoder()

        self.ReLU = nn.ReLU()

    def pack_latents(self, imgs, voxels):
        features = self.resnet(imgs)
        features = self.avgpool(features)
        features = features.view(-1, 24, 12, 12, 12)
        # print(features)
        # Nx1x128x128x128 --> Nx8x12x12x12
        voxelFeatures = self.encoder(voxels)

        # Nx32x12x12x12 --> Nx8x12x12x12
        combinedFeatures = torch.cat([voxelFeatures, features], dim=1)
        combinedFeatures = self.merger(combinedFeatures)
        return combinedFeatures

    def forward(self, imgs_1=None, voxels_1=None, imgs_2=None, voxels_2=None, lam=1):

        combinedFeatures_1 = self.pack_latents(imgs_1, voxels_1)
        combinedFeatures_2 = None
        if imgs_2!=None and voxels_2!=None:

            combinedFeatures_2 = self.pack_latents(imgs_2, voxels_2)
            mixup_features = lam.view(lam.shape[0], 1, 1, 1,1) * combinedFeatures_1 + (1 - lam).view(lam.shape[0],1,1,1,1) * combinedFeatures_2
        else:
            mixup_features = combinedFeatures_1
        # print(type(mixup_features))
        out = self.decoder(mixup_features.float())

        return combinedFeatures_1, combinedFeatures_2, mixup_features, out
