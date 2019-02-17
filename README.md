# CAE-ADMM: IMPLICIT BITRATE OPTIMIZATION VIA ADMM-BASED PRUNING IN COMPRESSIVE AUTOENCODERS

Haimeng Zhao, Peiyuan Liao



## Abstract

We introduce ADMM-pruned Compressive AutoEncoder (CAE-ADMM) that uses Alternative Direction Method of Multipliers (ADMM) to optimize the trade-off between distortion and efficiency of lossy image compression. Specifically, ADMM in our method is to promote sparsity to implicitly optimize the bitrate, different from entropy estimators used in the previous research. The experiments on public datasets show that our method outperforms the original CAE and some traditional codecs in terms of SSIM/MS-SSIM metrics, at reasonable inference speed.

## Paper
[arXiv:1901.07196 [cs.CV]](https://arxiv.org/abs/1901.07196)

## Model Architecture
![Model Architecture](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/model_new.jpg)

The architecture of CAE-ADMM. "Conv k/spP" stands for a convolutional layer with kernel size k times k with a stride of s and a reflection padding of P, and "Conv Down" is reducing the height and weight by 2.

## Performance
![](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/legend-new.png)
![](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/ssim.jpg) 
![](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/msssim.jpg)

Comparison of different method with respect to SSIM and MS-SSIM on the Kodak PhotoCD dataset. Note that Toderici et al. used RNN structure instead of entropy coding while CAE-ADMM (Ours) replaces entropy coding with pruning method.

## Example
![bpp 0.3](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/compare_03-new.jpg)
![bpp 0.3](https://raw.github.com/JasonZHM/CAE-ADMM/master/experiments/fig/latent.jpg)

Comparison of latent code before and after pruning for kodim21. For the sake of clarity, we marked zero values in the feature map before normalization as black.

## Citation
If you useIf you use these models in your research, please cite:
```
@article{zhao2019cae,
  title={CAE-ADMM: Implicit Bitrate Optimization via ADMM-based Pruning in Compressive Autoencoders},
  author={Zhao, Haimeng and Liao, Peiyuan},
  journal={arXiv preprint arXiv:1901.07196},
  year={2019}
}
```

## Acknowledgement
pytorch-msssim: Implementation of MS-SSIM in PyTorch is from [pytorch-msssim]( https://github.com/jorge-pessoa/pytorch-msssim)

huffmancoding.py: Implementation of Huffman coding is from [Deep-Compression-PyTorch](https://github.com/mightydeveloper/Deep-Compression-PyTorch)
