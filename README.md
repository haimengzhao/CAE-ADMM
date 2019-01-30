# CAE-ADMM: IMPLICIT BITRATE OPTIMIZATION VIA ADMM-BASED PRUNING IN COMPRESSIVE AUTOENCODERS

Haimeng Zhao, Peiyuan Liao



## Abstract

We introduce CAE-ADMM (ADMM-pruned compressive autoencoder), a lossy image compression model, inspired by researches in neural architecture search (NAS) and is capable of implicitly optimizing the bitrate without the use of an entropy estimator. Our experiments show that by introducing alternating direction method of multipliers (ADMM) to the model pipeline, the pruning paradigm yields more accurate results (SSIM/MS-SSIM-wise) when compared to entropy-based approaches and that of traditional codecs (JPEG, JPEG 2000, etc.) while maintaining acceptable inference speed. We further explore the effectiveness of the pruning method in CAE-ADMM by examining the generated latent codes.

## Paper (outdated)
[arXiv:1901.07196 [cs.CV]](https://arxiv.org/abs/1901.07196)

## Model Architecture
![Model Architecture](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/model_new.jpg)

## Performance
![](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/legend-new.jpg)
![](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/ssim.jpg) ![](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/msssim.jpg)

## Example
![bpp 0.3](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/compare_03-new.jpg)

## Acknowledgement
pytorch-msssim: Implementation of MS-SSIM in PyTorch is from [pytorch-msssim]( https://github.com/jorge-pessoa/pytorch-msssim)

huffmancoding.py: Implementation of Huffman coding is from [Deep-Compression-PyTorch](https://github.com/mightydeveloper/Deep-Compression-PyTorch)
