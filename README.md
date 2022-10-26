# Single-Shot Neural Relighting and SVBRDF Estimation ([Project](http://cseweb.ucsd.edu/~viscomp/projects/ECCV20NeuralRelighting/))


[Shen Sang](https://ssangx.github.io/), [Manmohan Chandraker](https://cseweb.ucsd.edu/~mkchandraker/)


## Overview

This is the official code release of our ECCV2020 paper "Single-Shot Neural Relighting and SVBRDF Estimation". Please consider citing this paper if you find the code and data useful in your project. Please contact us by ssang@eng.ucsd.edu if you have any questions or issues.



![TEASER](http://cseweb.ucsd.edu/~viscomp/projects/ECCV20NeuralRelighting/assets/teaser.png)



## Prerequisite
1. PyTorch with CUDA support
2. Python3



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


## Data preparation

### Download

Please download the synthetic dataset [here](https://drive.google.com/file/d/10bHDfrNPPcge8LqaOlLNyJ6IjGKkK6PF/view?usp=sharing). It contains all the materials and shape parameters (albedo, normal, roughness and depth) used for rendering. We also provide the script `rendering.py` for you to show how to render your own dataset. Unzip and rename it as `Synthetic`.


Make sure the structure is:

```
data
|-- datset
    |--Synthetic
        |--train
        |--test
|-- ...
```

### Pre-scan all files

Create the index file of all file names for the training set or test set by running `python dataset/make_pkl.py`.


## Training

1. Train the model for relighting under a single point light by running `python train_pt.py`.

2. Train the model for relighting under arbitrary environments and point light by running `python train_env.py`.


## Evaluation

1. Evaluate the trained model by running `python eval_env.py` and `python eval_pt.env`.
