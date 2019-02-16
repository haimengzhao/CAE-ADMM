# CAE-ADMM: IMPLICIT BITRATE OPTIMIZATION VIA ADMM-BASED PRUNING IN COMPRESSIVE AUTOENCODERS

Haimeng Zhao, Peiyuan Liao



## Abstract

We introduce ADMM-pruned Compressive AutoEncoder (CAE-ADMM) that uses Alternative Direction Method of Multipliers (ADMM) to optimize the trade-off between distortion and efficiency of lossy image compression. Specifically, ADMM in our method is to promote sparsity to implicitly optimize the bitrate, different from entropy estimators used in the previous research. The experiments on public datasets show that our method outperforms the original CAE and some traditional codecs in terms of SSIM/MS-SSIM metrics, at reasonable inference speed.

## Paper
[arXiv:1901.07196 [cs.CV]](https://arxiv.org/abs/1901.07196)

## Model Architecture
![Model Architecture](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/model_new.jpg)

## Performance
![](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/legend-new.png)
![](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/ssim.jpg) ![](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/msssim.jpg)

## Example
![bpp 0.3](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/compare_03-new.jpg)
![bpp 0.3](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/latent.jpg)

## Acknowledgement
pytorch-msssim: Implementation of MS-SSIM in PyTorch is from [pytorch-msssim]( https://github.com/jorge-pessoa/pytorch-msssim)

huffmancoding.py: Implementation of Huffman coding is from [Deep-Compression-PyTorch](https://github.com/mightydeveloper/Deep-Compression-PyTorch)
