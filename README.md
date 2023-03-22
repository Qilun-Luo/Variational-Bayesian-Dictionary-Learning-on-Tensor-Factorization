# Variational-Bayesian-Dictionary-Learning-on-Tensor-Factorization
Matlab Implementation for the paper:

Luo, Q., Li, W., & Xiao, M. (2023). Bayesian Dictionary Learning on Robust Tubal Transformed Tensor Factorization. IEEE Transactions on Neural Networks and Learning Systems. [DOI: 10.1109/TNNLS.2023.3248156](https://doi.org/10.1109/TNNLS.2023.3248156)

## Tensor RPCA
Suppose the third-order tensor $\mathcal{Y}\in \mathbb{R}^{n_1\times n_2\times n_3}$ is a noise measurement that admits a latent low-rank tensor $\mathcal{X}\in \mathbb{R}^{n_1\times n_2\times n_3}$ and a sparse tensor $\mathcal{S}$. That is,
$$\mathcal{Y} = \mathcal{X}+\mathcal{S}+\mathcal{E},$$
where $\mathcal{E}$ is the corrupted outliers, and $\mathcal{X} = (\mathcal{U}\ast_P \mathcal{V}^P) \times_3 D$ with the shared dictionary $D$.

To formulate the robust dictionary tubal decomposition over the Bayesian inference, a conditional distribution of the observed tensor $\mathcal{Y}$ is given as follows:

$$p(\mathcal{Y} | \mathcal{U}, \mathcal{V}, D, \mathcal{S}, \tau) = \prod_{i=1}^{n_1}\prod_{j=1}^{n_2} \mathcal{N}\left(\mathcal{Y}_{ij\cdot}|D(\mathcal{U}_{i,:,:} \ast_P \mathcal{V}_{j,:,:}^P)+\mathcal{S}_{ij\cdot},\; (\tau I_{n_3})^{-1}\right).$$

## Application
- Color Image Denoising: `demo_image_denoising.m`
- HSI Denoising: `demo_hsi_denoising.m`
- Background/Foreground Separation: `demo_background_subtraction.m`

**_Remark_**: **MATLAB 2020b or later is required**

 ## Folders
 - `algs` includes only the proposed method: VBDL.
 - `data` contains datasets.
    - `denoise_all`: includes only first six images of [Berkeley Segmentation Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/dataset/images.html) (BSD). Whole dataset **download** from [BSD](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/dataset/images.html)
      > Martin D, Fowlkes C, Tal D, et al. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics[C]//Proceedings Eighth IEEE International Conference on Computer Vision. ICCV 2001. IEEE, 2001, 2: 416-423.
    - `HSI`: [hyperspectral](https://rslab.ut.ac.ir/data) images;
    - `dataset2014`: **download** from [Google Drive](https://drive.google.com/drive/folders/1Nmy4AKNcmnpy-BiQIkK8VPyydR_mQQk0?usp=share_link) or [ChangeDetection.net(CDNet) dataset2014](http://changedetection.net/).
        > Wang Y, Jodoin P M, Porikli F, et al. CDnet 2014: An expanded change detection benchmark dataset[C]//Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2014: 387-394.
- `utils` includes some utilities scripts.
