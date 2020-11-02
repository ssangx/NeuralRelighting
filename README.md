# Single-Shot Neural Relighting and SVBRDF Estimation ([Project](http://cseweb.ucsd.edu/~viscomp/projects/ECCV20NeuralRelighting/))



[Shen Sang](https://sites.google.com/view/ssang), [Manmohan Chandraker](https://cseweb.ucsd.edu/~mkchandraker/)



Code release for our ECCV2020 paper "Single-Shot Neural Relighting and SVBRDF Estimation".


**This page is still under-construction. The currently released codes are used for inference with our pre-trained models and captured images.**



## Inference

Ensure that the folder structure under `data` is:

```
data
|-- models
|-- real
|-- output
|-- ...
```

Put your own test images under `real`. Then run `test_real_env.py` or `test_real_pt.py` to do inference. The estimated albedo, normal, roughness and depth, as well as the relighting images and videos will be shown under `data/output`.


## Data generation

You can download the synthetic dataset [here](https://drive.google.com/file/d/1kmgPzBhhZpozNA7QH2FOmfM51m6CKwNc/view?usp=sharing). It contains all the material and shape parameters (albedo, normal, roughness and depth) used for rendering. There are also image renderings included inside this dataset. We also provide the script `rendering.py` for you to render your own dataset. 


## Training

You can train the model for point light manipulation by yourself, by running 'train_pt.py`.
