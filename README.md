# Pose Adaptive Dual Mixup for Few-Shot Single-View 3D Reconstruction

This repository provides the source code for the paper [Pose Adaptive Dual Mixup for Few-Shot Single-View 3D Reconstruction](https://arxiv.org/abs/2112.12484?context=cs) published in AAAI-22. The implementation is on ShapeNet.

![Overview](https://github.com/ttchengab/PADMix/blob/main/overview.png)


## Cite this work

```
@inproceedings{padmix,
  title={Pose Adaptive Dual Mixup for Few-Shot Single-View 3D Reconstruction},
  author={Cheng, Ta-Ying and 
          Yang, Hsuan-Ru and 
          Trigoni, Niki and 
          Chen, Hwann-Tzong and 
          Liu, Tyng-Luh},
  booktitle={AAAI},
  year={2022}
}
```


## Prerequisites

#### Clone the Code Repository

```
git clone https://github.com/ttchengab/PADMix.git
```

#### Datasets


The [ShapeNet](https://www.shapenet.org/) dataset is available below:

- ShapeNet rendering images: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
- ShapeNet voxelized models: http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz

## Get Started

Create two directories, one for saving templates, and the other for saving checkpoints:

```
mkdir template
mkdir ckpts_fewshot
```

To generate the priors, use the following command:

```
python saveTV.py
```

To train the ground truth autoencoder, use the following command:

```
python fewshot_AE.py
```

To train with Input Mixup, use the following command:

```
python fewshot_mixup_triplet.py
```

To train with Latent Mixup, use the following command:

```
python fewshot_latent_mixup.py
```

To evaluate the IoU, use the following command:

```
python fewshot_eval.py
```

## Pretrained Models

The pretrained models under the one-shot setting on ShapeNet are available [here](https://drive.google.com/file/d/1BMnLMIHyhCTlBN5w6T2iOa8rHwmcVajt/view?usp=sharing)
