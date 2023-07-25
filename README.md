
# Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation

 The source code of our work **"Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation**
![img|center](./img/pipeline.png)

## News
- **2022-06 [NEW:fire:]** **PVKD (CVPR2022)**, a lightweight Cylinder3D model with much higher performance has been released [here](https://github.com/cardwing/Codes-for-PVKD)
-  Cylinder3D is accepted to CVPR 2021 as an **Oral** presentation
-  Cylinder3D achieves the **1st place** in the leaderboard of SemanticKITTI **multiscan** semantic segmentation
<p align="center">
   <img src="./img/leaderboard2.png" width="30%"> 
</p>

- Cylinder3D achieves the 2nd place in the challenge of nuScenes LiDAR segmentation, with mIoU=0.779, fwIoU=0.899 and FPS=10Hz.
- **2020-12** We release the new version of Cylinder3D with nuScenes dataset support.
- **2020-11** We preliminarily release the Cylinder3D--v0.1, supporting the LiDAR semantic segmentation on SemanticKITTI and nuScenes.
- **2020-11** Our work achieves the **1st place** in the leaderboard of SemanticKITTI semantic segmentation (until CVPR2021 DDL, still rank 1st in term of Accuracy now), and based on the proposed method, we also achieve the **1st place** in the leaderboard of SemanticKITTI panoptic segmentation.

<p align="center">
   <img src="./img/leaderboard.png" width="40%"> 
</p>

## Installation

### Requirements
- PyTorch >= 1.2 
- yaml
- Cython
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)   done
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) (optional for nuScenes)
- [spconv](https://github.com/traveller59/spconv) (tested with spconv==1.2.1 and cuda==10.2)  done  

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple strictyaml
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple spconv-cu117
conda install pytorch-scatter -c pyg
```

依赖库都存在，可以用于plugin集成
spconv的C++实现版本
pytorch_scatter的C++实现

## Data Preparation

### SemanticKITTI
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

### nuScenes
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
		├──v1.0-trainval
		├──v1.0-test
		├──samples
		├──sweeps
		├──maps

```

## Training
1. modify the config/semantickitti.yaml with your custom settings. We provide a sample yaml for SemanticKITTI
2. train the network by running "sh train.sh"

### Training for nuScenes
Please refer to [NUSCENES-GUIDE](./NUSCENES-GUIDE.md)

### Pretrained Models
-- We provide a pretrained model for SemanticKITTI [LINK1](https://drive.google.com/file/d/1q4u3LlQXz89LqYW3orXL5oTs_4R2eS8P/view?usp=sharing) or [LINK2](https://pan.baidu.com/s/1c0oIL2QTTcjCo9ZEtvOIvA) (access code: xqmi)

-- For nuScenes dataset, please refer to [NUSCENES-GUIDE](./NUSCENES-GUIDE.md)

## Semantic segmentation demo for a folder of lidar scans
```
python demo_folder.py --demo-folder YOUR_FOLDER --save-folder YOUR_SAVE_FOLDER
```
If you want to validate with your own datasets, you need to provide labels.
--demo-label-folder is optional
```
python demo_folder.py --demo-folder YOUR_FOLDER --save-folder YOUR_SAVE_FOLDER --demo-label-folder YOUR_LABEL_FOLDER
```

## TODO List
- [x] Release pretrained model for nuScenes.
- [x] Support multiscan semantic segmentation.
- [ ] Support more models, including PolarNet, RandLA, SequeezeV3 and etc.
- [ ] Integrate LiDAR Panotic Segmentation into the codebase.

## Reference

If you find our work useful in your research, please consider citing our [paper](https://arxiv.org/pdf/2011.10033):
```
@article{zhu2020cylindrical,
  title={Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation},
  author={Zhu, Xinge and Zhou, Hui and Wang, Tai and Hong, Fangzhou and Ma, Yuexin and Li, Wei and Li, Hongsheng and Lin, Dahua},
  journal={arXiv preprint arXiv:2011.10033},
  year={2020}
}

#for LiDAR panoptic segmentation
@article{hong2020lidar,
  title={LiDAR-based Panoptic Segmentation via Dynamic Shifting Network},
  author={Hong, Fangzhou and Zhou, Hui and Zhu, Xinge and Li, Hongsheng and Liu, Ziwei},
  journal={arXiv preprint arXiv:2011.11964},
  year={2020}
}
```

## Acknowledgments
We thanks for the opensource codebases, [PolarSeg](https://github.com/edwardzhou130/PolarSeg) and [spconv](https://github.com/traveller59/spconv)


下载seg-kitti数据集，下载cylinder预训练模型

## 激光雷达语义分割:
### 投影到2D空间
球面/BEV投影 spherical/BEV projection  2D图像损失3D拓扑和几何关系
1. 球面投影得到密集2D图像
2. 鸟瞰图像，压缩高度信息
3. 体素化， 长方体，圆柱体

### 直接在3D空间处理点云
1. 3D cylinder partition and a 3D cylinder convolution
3D圆柱划分和3D圆柱卷积， 3D网络
- __3维卷积__  提取3D特征 对体素化后的3D网格进行3D卷积
- __圆柱分区处理__ 平衡驾驶场景点云按远近分布不均的特点
- 长方体物体使用里非对称残差模块

2. 维度分解的上下文建模，融合多帧信息

### 网络结构
1. 点云圆柱体分区(3D 表示)
cylinder_fea: 
  1. input :pt_fea_ten: [60000, 9], grid_ten: [60000, 3]
  2. 对grid_ten进行pad  [60000, 9],           [60000, 4]


Asymm_3d_spconv:


2. 3D U-Net 处理3D表示 输入是[C, H, W, L]
- 非对称残差模块来适应长方体物体 asymmetry residual block
- 基于维度分解的上下文建模 Dimension-Decomposition Based Context Modeling
3. 分割头， segmentation backbone 是 3d卷积层(kernel 3x3x3) 输出是[Class, H, W, L]

主要就是Conv3D 和 DeConv3D

圆柱坐标系代替笛卡尔坐标系
- 坐标系(x, y, z)转换到柱坐标(p, e, z)，随距离的增加体素增大。送入3D点网络，得到特征图为[C, H, W, L] C为特征维度
- 基于维度分解的上下文建模(DDCM), 分为长，宽，高共3个维度的, 之后进行融合

主干网络是U-Net， 3D卷积来自spconv

4. loss
For network optimization, we use __a weighted cross-entropy loss__ and __a lovasz-softmax loss__ to
maximize the point accuracy and the intersection-over-union score for classes. Two losses share the same weight. Thus, the total loss is: ζ all = ζ iou + ζ acc . For the optimizer, Adam with an initial learning rate of 0.001, is employed.

5. dataset
SemanticKITTI, 原始数据集共28个类别，最终保留19个类别

### infer_test
```
python demo_folder.py -y ./config/semantickitti.yaml --demo-folder ./work/infer_test/velodyne --save-folder ./work/infer_test/labels/
```
1650 显存占用
3861MiB /  3911MiB
1650 可以推理，将代码适配spconv1 => spconv2


### train
```
bash train.sh
```
- 1650模型训练显存不足  单batch也不行， 导出onnx也会显存不足
- 4060可以训练，   7616MiB /  8188MiB



数据读取与预处理
xxx.bin文件中存储的是点的坐标和强度值
(498420,)
(124605, 4)
```
self.im_idx[index]: /home/br/program/cylinder3d/work/infer_test/velodyne/000000.bin
raw_data.shape: (124668, 4)

self.im_idx[index]: /home/br/program/cylinder3d/work/infer_test/velodyne/000001.bin
raw_data.shape: (124605, 4)

self.im_idx[index]: /home/br/program/cylinder3d/work/infer_test/velodyne/000002.bin
raw_data.shape: (124478, 4)

self.im_idx[index]: /home/br/program/cylinder3d/work/infer_test/velodyne/000003.bin
raw_data.shape: (124167, 4)

self.im_idx[index]: /home/br/program/cylinder3d/work/infer_test/velodyne/000004.bin
raw_data.shape: (123969, 4)
```

需要在新电脑安装环境
1. pytorch支持CUDA11.7 CUDA11.8等
pytorch2.0/1.13 支持CUDA11.7 因此安装cuda11.7
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
2. CUDA11.7支持的Ubuntu版本为
(https://docs.nvidia.com/cuda/archive/11.7.1/cuda-installation-guide-linux/index.html)
Distribution              Kernel      DefaultGCC  GLIBC  
Ubuntu 22.04 	            5.15.0-25 	11.2.0 	    2.35  
Ubuntu 20.04.z (z <= 4) 	5.13.0-30 	9.3.0 	    2.31  
Ubuntu 18.04.z (z <= 6) 	5.4.0-89 	  7.5.0 	    2.27  
0606下载的版本是20.04.6不满足要求, 而且据调查，ubuntu20.04存在新网卡，声卡等硬件兼容问题，需要升级内核版本到(5.14)，而升级后无法和CUDA安装需求对应，因此安装ubuntu22.04


WSL2也可以用cuda？
https://zhuanlan.zhihu.com/p/621142457
https://learn.microsoft.com/en-us/windows/wsl/install
https://developer.nvidia.com/cuda/wsl
https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl 
目前WSL不支持Unified Memory和Pinned system memory, Root user on bare metal (not containers) will not find nvidia-smi at the expected location.等限制
还是算了，安装ubuntu22.04

### 导出onnx
导出pretrained model为onnx
cylinder3d_pretrained_model.pt(214M)  ==>  cylinder3d_pretrained.onnx(5.1M)

onnx spconv的算子导出 **TODO**
https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/issues/40
#### spconv
(spconv)[https://zhuanlan.zhihu.com/p/467167809]



回去重新在kitti数据集下预训练模型上fine tune
为何预训练模型的输出类别都是70？？