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

## Visulization on CIFAR-10

A-STN

![](./vis/STN.gif)

D-VAE

![](./vis/deform.gif)

P-VAE

![](./vis/deform.gif)


