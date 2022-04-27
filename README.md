# GFS-Seg
This is the implementation of [**Generalized Few-shot Semantic Segmentation**](https://arxiv.org/abs/2010.05210) (CVPR 2022). 

![image](https://user-images.githubusercontent.com/68939582/160376201-2bc953b6-280e-4cb8-b512-31ffa3a3e579.png)


# Get Started

### Environment
+ Python 3.7.9
+ Torch 1.5.1
+ cv2 4.4.0
+ numpy 1.21.0
+ CUDA 10.1

### Datasets and Data Preparation
Different from [**PFENet**](https://github.com/dvlab-research/PFENet) (5953 images), in GFS-Seg, the training set of Pascal-VOC is used with augmented data (10582 images), following the original [**PSPNet**](https://github.com/hszhao/semseg). But our experiments in FS-Seg are yieled follow the setting of PFENet.

The preparation of [**COCO 2014**](https://cocodataset.org/#download) follows [**PFENet**](https://github.com/dvlab-research/PFENet). 

This code reads data from .txt files where each line contains the paths for image and the correcponding label respectively. Image and label paths are seperated by a space. Example is as follows:

    image_path_1 label_path_1
    image_path_2 label_path_2
    image_path_3 label_path_3
    ...
    image_path_n label_path_n

Then update the train/val list paths in the config files.

### Run Demo / Test with Pretrained Models
+ Execute `mkdir initmodel` at the root directory.
+ Download the ImageNet pretrained [**backbones**](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EQEY0JxITwVHisdVzusEqNUBNsf1CT8MsALdahUhaHrhlw?e=4%3a2o3XTL&at=9) and put them into the `initmodel` directory.
+ Please download the pretrained models.
+ We provide **16 pre-trained**  [**models**](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155122171_link_cuhk_edu_hk/Ej4c5aUV1RxDpXuSjK-9BZYBKF23mgq2zR8bNYWTkIJtkA?e=Rhb1xi): 
8 for 1/5 shot results on PASCAL-5i and 8 for COCO.
+ Update the config files by speficifying the target **split**, **weights** and **shot** for loading different checkpoints.

Note: The pre-trained models are re-trained by this repo, and you should be able to get generally comparable or slightly better performance than the ones reported in our paper.


### Train / Evaluate
+ For training, please set the option **only_evaluate** to **False** in the configuration file. Then execute this command at the root directory: 

    sh train.sh {*dataset*} {*model_config*}
    
+ For evaluation only, please set the option **only_evaluate** to **True** in the corresponding configuration file. 

    
Example: Train / Evaluate CAPL with 1-shot on the split 0 of PASCAL-5i: 

    sh train.sh pascal split0_1shot   
    
### Implementation in FS-Seg
We have provided our implementation of CAPL in the setting of classic few-shot segmentation:

https://github.com/tianzhuotao/CAPL-FSSeg
    
    
## Related Assets \& Acknowledgement

Our work is closely related to the following assets that inspire our implementation. We gratefully thank the authors. 
+ RePRI: https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation
+ HSNet: https://github.com/juhongm999/hsnet
+ SCL: https://github.com/zbf1991/SCL
+ ASGNet: https://github.com/Reagan1311/ASGNet
+ CANet: https://github.com/icoz69/CaNet
+ PANet: https://github.com/kaixin96/PANet
+ PPNet: https://github.com/Xiangyi1996/PPNet-PyTorch
+ PFENet: https://github.com/dvlab-research/PFENet
+ SemSeg: https://github.com/hszhao/semseg

# Citation

If you find this project useful, please consider citing:
```
@InProceedings{tian2022gfsseg,
    title={Generalized Few-shot Semantic Segmentation},
    author={Zhuotao Tian and Xin Lai and Li Jiang and Shu Liu and Michelle Shu and Hengshuang Zhao and Jiaya Jia},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```
