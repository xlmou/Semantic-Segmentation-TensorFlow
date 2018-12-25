# Semantic Segmentation Model in TensorFlow

## Description
This repository contains some sementic segmentation models implemented by tensorflow and the pipeline of training and evaluating models as follows:
- Convert dataset to TFRecord files format.
- Data augmentation (e.g., scaling, padding, cropping and flipping).
- Training various segmentation models with different datasets.
- Evaluting trained models with mean Intersection over Union (mIoU).

## Models
- FCN8: [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- U-Net: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- SegNet: [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561)
- Deeplab-v1: [Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs](https://arxiv.org/abs/1412.7062)
- Deeplab-v2: [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)
- Deeplab-v3: [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
- PSPNet: [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
- GCN: [Large Kernel Matters——Improve Semantic Segmentation by Global Convolutional Network](https://arxiv.org/abs/1703.02719)
- ENet: [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147)
- ICNet: [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545)

## Dataset preparing
Before training a semantic segmentation model, the dataset should be downloaded and converted to TFRecord files. 
- Download datasets([pascal_voc2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html), [cityscapes](https://www.cityscapes-dataset.com/dataset-overview/#features), [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/), ......)
Then split the dataset into four partitions (training images, training labels, validation images and validation labels ) and put them in separate folders.
- Convert dataset to TFRecord files:
```
cd dataset

python build_TFRecord.py --train-image-path folder_name --train-label-path [folder_name] --val-image-path [folder_name] --val-label-path [folder_name]
```

## Training with pascal_voc2012 or other dataset...
Having prepared the dataset of TFRecord file format, the pretrained weights of both [vgg_16](https://github.com/tensorflow/models/tree/master/research/slim) and [resnet_v2_101](https://github.com/tensorflow/models/tree/master/research/slim) should be downloaded and restored in semantic segmentation model. The folders should be set up in following structure:

    ├── pretrained                   
    |   ├── vgg_16.ckpt
    |   ├── resnet_v2_101.ckpt

Training shell script (using pascal_voc2012):
- FCN8: `python train.py --data-dir './dataset/tfrecord' --restore-from './pretrained/vgg_16.ckpt' --dataset 'pascal_voc2012' --pretrained-model 'vgg_16' --segmentation-model 'FCN8'`
- U-Net: `python train.py --data-dir './dataset/tfrecord' --dataset 'pascal_voc2012' --segmentation-model 'U_Net'`
- SegNet: `python train.py --data-dir './dataset/tfrecord' --dataset 'pascal_voc2012' --segmentation-model 'Seg_Net'`
- Deeplab-v1: `python train.py --data-dir './dataset/tfrecord' --restore-from './pretrained/vgg_16.ckpt' --dataset 'pascal_voc2012' --pretrained-model 'vgg_16' --segmentation-model 'Deeplab_v1'`
- Deeplab-v2: `python train.py --data-dir './dataset/tfrecord' --restore-from './pretrained/resnet_v2_101.ckpt' --dataset 'pascal_voc2012' --pretrained-model 'resnet_v2_101' --segmentation-model 'Deeplab_v2'`
- Deeplab-v3: `python train.py --data-dir './dataset/tfrecord' --restore-from './pretrained/resnet_v2_101.ckpt' --dataset 'pascal_voc2012' --pretrained-model 'resnet_v2_101' --segmentation-model 'Deeplab_v3'`
- PSPNet: `python train.py --data-dir './dataset/tfrecord' --restore-from './pretrained/resnet_v2_101.ckpt' --dataset 'pascal_voc2012' --pretrained-model 'resnet_v2_101' --segmentation-model 'PSPNet'`
- GCN: `python train.py --data-dir './dataset/tfrecord' --restore-from './pretrained/resnet_v2_101.ckpt' --dataset 'pascal_voc2012' --pretrained-model 'resnet_v2_101' --segmentation-model 'GCN'`
- ENet: `python train.py --data-dir './dataset/tfrecord' --dataset 'pascal_voc2012' --segmentation-model 'ENet'`
- ICNet: `python train.py --data-dir './dataset/tfrecord' --dataset 'pascal_voc2012' --segmentation-model 'ICNet'`

