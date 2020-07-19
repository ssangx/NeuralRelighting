# Single-Shot Neural Relighting and SVBRDF Estimation ([Project](http://cseweb.ucsd.edu/~viscomp/projects/ECCV20NeuralRelighting/))



[Shen Sang](https://sites.google.com/view/ssang), [Manmohan Chandraker](https://cseweb.ucsd.edu/~mkchandraker/)



Code release for our ECCV2020 paper "Single-Shot Neural Relighting and SVBRDFEstimation".


**This page is still under-construction. The currently released codes are used for inference with our pre-trained models and captured images.**



## Inference

Ensure that the folder structure under `data` is:

```
data
|-- dataset
|-- models
|-- output
|-- real
```

Put your own test images under `real`. Then run `test_real_env.py` or `test_real_pt.py` to do inference. The estimated albedo, normal, roughness and depth, as well as the relighting images and videos will be shown under 'data/output'.



## Training

Coming soon
