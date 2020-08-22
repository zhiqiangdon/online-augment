# OnlineAugment (Accepted at ECCV 2020)

Official [OnlineAugment](https://arxiv.org/abs/2007.09271) implementation in PyTorch

- More automatic than AutoAugment and related 
  - Towards fully automatic (STN and VAE, No need to specify the image primitives). 
  - Broad domains (natural, medical images, etc). 
  - Diverse tasks (classification, segmentation, etc). 
- Easy to use 
  - One-stage training (user-friendly). 
  - Simple code (single GPU training, no need for parallel optimization). 
- Orthogonal to AutoAugment and related 
  - Online v.s. Offline (Joint optimization, no expensive offline policy searching). 
  - State-of-the-art performance (in combination with AutoAugment). 
  
 ![](./vis/framework.png)
 
 (In this implementation, we disable the meta-gradient for efficient training. The code is also refactored accordingly, achieving comparable performance. Especially for reduced CIFARs, we observe higher accuracy than the paper results.)

## Visualization on CIFAR-10

A-STN

![](./vis/STN.gif)

D-VAE

![](./vis/deform.gif)

P-VAE

![](./vis/texture.gif)

## Run
We conducted experiments in
- python 3.7
- pytorch 1.2, torchvision 0.4.0, cuda10

The searching of policies and training of target model is optimized jointly. 

For example, training wide-resnet using STN on reduced CIFAR-10, using the script in r-cifar10-wrn-scripts

```
./run-aug-stn.sh
```


## Citation
If this code is helpful for your research, please cite:

```
@article{tang2020onlineaugment,
  title={OnlineAugment: Online Data Augmentation with Less Domain Knowledge},
  author={Tang, Zhiqiang and Gao, Yunhe and Karlinsky, Leonid and Sattigeri, Prasanna and Feris, Rogerio and Metaxas, Dimitris},
  journal={arXiv preprint arXiv:2007.09271},
  year={2020}
}
```

