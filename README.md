# Content-biased and Style-Assisted Transfer Network for Cross-scene Hyperspectral Image Classification

## Requirements
ConfigArgParse

scikit-learn

matplotlib

xlwt

pyyaml

Python 3.9

Pytorch 1.4

## Dataset
The dataset directory should look like this:
```
datasets
├── Pavia
│   ├── paviaC.mat
│   ├── paviaC_7gt.mat
│   ├── paviaU.mat
│   └── paviaU_7gt.mat
├── Houston
│   ├── Houston13.mat
│   ├── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
└── HyRANK
    ├── Dioni.mat
    ├── Dioni_gt_out68.mat
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

3. Run
```
./train.sh
```
* You can replace the training dataset/task by modifying `train.sh`.
* Note the modification of the parameters in `train.sh`.