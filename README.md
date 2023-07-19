## Introduction

### A Gated Cross-domain Collaborative Network for Underwater Object Detection (https://arxiv.org/pdf/2306.14141.pdf).
Underwater object detection (UOD) plays a significant role in aquaculture and marine environmental protection. Considering the challenges posed by low contrast and low-light conditions in underwater environments, several underwater image enhancement (UIE) methods have been proposed to improve the quality of underwater images. However, only using the enhanced images does not improve the performance of UOD, since it may unavoidably remove or alter critical patterns and details of underwater objects. In contrast, we believe that exploring the complementary information from the two domains is beneficial for UOD. The raw image preserves the natural characteristics of the scene and texture information of the objects, while the enhanced image improves the visibility of underwater objects. Based on this perspective, we propose a Gated Cross-domain Collaborative Network (GCC-Net) to address the challenges of poor visibility and low contrast in underwater environments, which comprises three dedicated components. Firstly, a real-time UIE method is employed to generate enhanced images, which can improve the visibility of objects in low-contrast areas. Secondly, a cross-domain feature interaction module is introduced to facilitate the interaction and mine complementary information between raw and enhanced image features. Thirdly, to prevent the contamination of unreliable generated results, a gated feature fusion module is proposed to adaptively control the fusion ratio of cross-domain information. Our method presents a new UOD paradigm from the perspective of cross-domain information interaction and fusion. Experimental results demonstrate that the proposed GCC-Net achieves state-of-the-art performance on four underwater datasets. 
<img width="971" alt="image" src="https://github.com/Ixiaohuihuihui/GCC-Net/assets/26215859/b3b66574-b23d-4ea7-9eb9-cc868800acd9">
<img width="977" alt="image" src="https://github.com/Ixiaohuihuihui/GCC-Net/assets/26215859/99665b78-69c8-4435-9b32-bb9556ed6c5b">

### Download Dataset
#### (1) DUO dataset (https://github.com/chongweiliu/DUO)
#### (2) Trashcan dataset (https://conservancy.umn.edu/handle/11299/214865)
#### (3) WPBB dataset (https://github.com/fedezocco/MoreEffEffDetsAndWPBB-TensorFlow/tree/main/WPBB_dataset)
#### (4) Brackish dataset (https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset)

### Modify the dataset path in your project 
for example, the path in 'configs'.

### Train

#### 1. For DUO dataset
```   
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/autoassign/autoassign_r50_fpn_8x2_3x_gcc_duo.py 2
```
#### 2. For TrashCan dataset
```   
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/autoassign/autoassign_r50_fpn_8x2_3x_gcc_trashcan.py 2
```
#### 3. For WPBB dataset
```   
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/autoassign/autoassign_r50_fpn_8x2_3x_gcc_wpbb.py 2
```
#### 4. For Brackish dataset
```   
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/autoassign/autoassign_r50_fpn_8x2_3x_gcc_wpbb.py 2
```

### Test
For main results, I have provided the currently trained weight file, which can be downloaded directly for testing.
```
CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/autoassign/autoassign_r50_fpn_8x2_3x_gcc_duo.py weights/gcc_net_epoch_36.pth --eval bbox
```
### Some Results
<img width="917" alt="image" src="https://github.com/Ixiaohuihuihui/GCC-Net/assets/26215859/03257154-4ce3-49a4-99d5-ac7972353381">
<img width="976" alt="image" src="https://github.com/Ixiaohuihuihui/GCC-Net/assets/26215859/664f8989-0f52-4f26-bd25-db333ea9bb63">

## Installation

Please refer to [install.md](docs/en/install.md) for installation guide.

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors

