# 5472_final_varbvs
The implementation of BVSR model by VA-EM for 5472 final project. 
### Description
The model is like a sparse version of the LMM model, which assumes "spike-slap" prior for coefficient instead of normal prior in LMM.
$$\beta\sim \pi * N(0, \sigma^2) + (1-\pi)*\delta_0$$
It is fitted by a variational expectation-maximization algorithm detailed in “varbvs: Fast Variable Selection for Large-scale Regression”.

Here I only implement the linear regression part of it.
### Reference
http://pcarbo.github.io/varbvs

Carbonetto, P., Zhou, X., & Stephens, M. (2017). varbvs: Fast Variable Selection for Large-scale Regression. arXiv preprint arXiv:1709.06597.

