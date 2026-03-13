# Content-biased and Style-Assisted Transfer Network for Cross-scene Hyperspectral Image Classification
Code for paper:
[Content-Biased and Style-Assisted Transfer Network for Cross-Scene Hyperspectral Image Classification](https://ieeexplore.ieee.org/abstract/document/10678753)

## Abstract
Cross-scene hyperspectral image (HSI) classification remains a challenging task due to the distribution discrepancies that arise from variations in imaging sensors, geographic regions, atmospheric conditions, and other factors between the source and target domains. Recent research indicates that convolutional neural networks (CNNs) exhibit a significant tendency to prioritize image styles, which are highly sensitive to domain variations, over the actual content of the images. However, few existing domain adaptation (DA) methods for cross-scene HSI classification take into consideration the style variations both within the samples of an HSI and between the cross-scene source and target domains. Accordingly, we propose a novel content-biased and style-assisted transfer network (CSTnet) for unsupervised DA (UDA) in cross-scene HSI classification. The CSTnet introduces a content and style reorganization (CSR) module that disentangles content features from style features via instance normalization (IN), while refining useful style information as a complementary component to enhance discriminability. A contentwise reorganization loss is designed to reduce the disparity between the separated content/style representations and the output features, thereby enhancing content-level alignment across different domains. Furthermore, we incorporate batch nuclear-norm maximization (BNM) as an effective class-balancing technique that directly exploits unlabeled target data to enhance minority class representations without requiring prior knowledge or pseudolabels, achieving better distribution alignment. Comprehensive experiments on three cross-scene HSI datasets demonstrate that the proposed CSTnet achieves state-of-the-art performance, effectively leveraging content bias and style assistance for robust DA in cross-scene HSI classification tasks.

## Methods' framework
<p align='center'>
  <img src='figure/overview.png' width="900px">
</p>

## Paper
Please cite our paper if you find the code or dataset useful for your research.
```
@ARTICLE{10678753,
  author={Shi, Zuowei and Lai, Xudong and Deng, Juan and Liu, Jinshuo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Content-Biased and Style-Assisted Transfer Network for Cross-Scene Hyperspectral Image Classification}, 
  year={2024},
  volume={62},
  number={},
  pages={1-17},
  keywords={Feature extraction;Shape;Hyperspectral imaging;Semantics;Predictive models;Degradation;Training;Class imbalance;content feature;cross scene;domain adaption (DA);hyperspectral image (HSI) classification},
  doi={10.1109/TGRS.2024.3458014}}
```

## Requirements
ConfigArgParse

scikit-learn

matplotlib

xlwt

pyyaml

Python 3.9

torch 1.12.1

CUDA Version 11.6

## Dataset
The dataset directory should look like this:
```
datasets
├── Pavia
│   ├── Source
│   │   ├── paviaU.mat
│   │   └── paviaU_7gt.mat
│   └── Target
│       ├── paviaC.mat
│       └── paviaC_7gt.mat
├── Houston
│   ├── Source
│   │   ├── Houston13.mat
│   │   └── Houston13_7gt.mat
│   └── Target
│       ├── Houston18.mat
│       └── Houston18_7gt.mat
└── HyRANK
    ├── Source
    │   ├── Dioni.mat
    │   └── Dioni_gt_out68.mat
    └── Target
        ├── Loukia.mat
        └── Loukia_gt_out68.mat
```

## Usage
1. Clone the repository
```
git clone https://github.com/nbdszw/CSTnet.git
```

2. Install the dependencies
```
pip install -r requirements.txt
```

3. Download [datasets](https://github.com/YuxiangZhang-BIT/Data-CSHSI) here.


4. Run
```
./train.sh
```
* You can replace the training dataset/task by modifying `train.sh`.
* Note the modification of the parameters in `train.sh`.