
## Introduction

### A Gated Cross-domain Collaborative Network for Underwater Object Detection (https://arxiv.org/pdf/2306.14141.pdf).
Underwater object detection (UOD) plays a significant role in aquaculture and marine environmental protection. Considering the challenges posed by low contrast and low-light conditions in underwater environments, several underwater image enhancement (UIE) methods have been proposed to improve the quality of underwater images. However, only using the enhanced images does not improve the performance of UOD, since it may unavoidably remove or alter critical patterns and details of underwater objects. In contrast, we believe that exploring the complementary information from the two domains is beneficial for UOD. The raw image preserves the natural characteristics of the scene and texture information of the objects, while the enhanced image improves the visibility of underwater objects. Based on this perspective, we propose a Gated Cross-domain Collaborative Network (GCC-Net) to address the challenges of poor visibility and low contrast in underwater environments, which comprises three dedicated components. Firstly, a real-time UIE method is employed to generate enhanced images, which can improve the visibility of objects in low-contrast areas. Secondly, a cross-domain feature interaction module is introduced to facilitate the interaction and mine complementary information between raw and enhanced image features. Thirdly, to prevent the contamination of unreliable generated results, a gated feature fusion module is proposed to adaptively control the fusion ratio of cross-domain information. Our method presents a new UOD paradigm from the perspective of cross-domain information interaction and fusion. Experimental results demonstrate that the proposed GCC-Net achieves state-of-the-art performance on four underwater datasets. 
![Snipaste_2022-06-17_11-58-45](https://user-images.githubusercontent.com/26215859/174222183-2de9fe00-8dd2-4535-8427-d9c385f145f8.png)
<img width="474" alt="image" src="https://user-images.githubusercontent.com/26215859/192183273-e86ee8f0-e96e-4251-a4c3-20885cb497f9.png">

### Download Dataset
#### DUO dataset (https://github.com/chongweiliu/DUO)
#### Trashcan dataset (https://conservancy.umn.edu/handle/11299/214865)
#### WPBB dataset (https://github.com/fedezocco/MoreEffEffDetsAndWPBB-TensorFlow/tree/main/WPBB_dataset)
#### Brackish dataset (https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset)

### Modify the dataset path in your project 
for example, the path in 'configs'.

### Train

#### For DUO dataset
```   
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/autoassign/autoassign_r50_fpn_8x2_3x_gcc_duo.py 2
```
#### For TrashCan dataset
```   
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/autoassign/autoassign_r50_fpn_8x2_3x_gcc_trashcan.py 2
```
#### For WPBB dataset
```   
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/autoassign/autoassign_r50_fpn_8x2_3x_gcc_wpbb.py 2
```

#### For Brackish dataset
```   
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/autoassign/autoassign_r50_fpn_8x2_3x_gcc_wpbb.py 2
```

### Test

```
CUDA_VISIBLE_DEVICES=5 ./tools/dist_test.sh configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_dota.py work_dirs/deformable_detr_twostage_refine_r50_16x2_50e_dota/epoch_50.pth 1 --format-only --eval-options submission_dir=work_dirs/deformable_detr_twostage_refine_r50_16x2_50e_dota/Task1_results 
```
### Some Results
![image](https://user-images.githubusercontent.com/26215859/174222334-df51f640-c267-4f1e-a9e4-25edd2b9eee1.png)

![image](https://user-images.githubusercontent.com/26215859/174222294-68698a0b-8d82-41c0-8c02-a2aa182f8e42.png)


## Installation

Please refer to [install.md](docs/en/install.md) for installation guide.

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.

## Data Preparation

Please refer to [data_preparation.md](tools/data/README.md) to prepare the data.

## Acknowledgement

MMRotate is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

