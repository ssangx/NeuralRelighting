# Single-Shot Neural Relighting and SVBRDF Estimation ([Project](http://cseweb.ucsd.edu/~viscomp/projects/ECCV20NeuralRelighting/))


[Shen Sang](https://sites.google.com/view/ssang), [Manmohan Chandraker](https://cseweb.ucsd.edu/~mkchandraker/)


## Overview

This is the official code release of our ECCV2020 paper "Single-Shot Neural Relighting and SVBRDF Estimation". Please consider citing this paper if you find the code and data useful in your project. Please contact us by ssang@eng.ucsd.edu if you have any questions or issues.



![TEASER](http://cseweb.ucsd.edu/~viscomp/projects/ECCV20NeuralRelighting/assets/teaser.png)



## Prerequisite
1. PyTorch with CUDA support



## Test on real image

We have included the pretrained models and some test cases inside this repo. Ensure that the folder structure under `data` is:

```
data
|-- models
|-- real
|-- output
|-- ...
```

Put your own test images under `real`. Then run `test_real_env.py` or `test_real_pt.py` to do inference. The estimated albedo, normal, roughness and depth, as well as the relighting images and videos will be shown under `data/output`.


## Data prepareation

### 1. Relighting under a single point light
Please download the synthetic dataset [here](https://drive.google.com/file/d/1kmgPzBhhZpozNA7QH2FOmfM51m6CKwNc/view?usp=sharing). It contains all the materials and shape parameters (albedo, normal, roughness and depth) used for rendering. There are also image renderings included inside this dataset. We also provide the script `rendering.py` for you to render your own dataset. Unzip and rename it as 'SyntheticPt'.


### 2. Relighting under a point light + arbitray environments
Please download the synthetic dataset [here](http://cseweb.ucsd.edu/~viscomp/projects/SIGA18ShapeSVBRDF/Data.zip). Unzip and rename it as 'SyntheticEnv'.


Make sure the structure is:

```
data
|-- datset
    |--SyntheticPt
    |--SyntheticEnv
|-- ...
```

These two datasets are used for the two different relighting tasks. For any single task, you do not need to download both of them, please download the one that corresponding to your demands.


## Training

1. Train the model for relighting under a single point light by running `train_pt.py`.

2. Train the model for relighting under arbitrary environments and point light by running `train_env.py`.
