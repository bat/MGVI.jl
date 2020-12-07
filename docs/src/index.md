# MGVI.jl

MGVI is an iterative method that performs a series of Gaussian approximations to the posterior. We alternate between approximating the covariance with the inverse Fisher information metric evaluated at an intermediate mean estimate and optimizing the KL-divergence for the given covariance with respect to the mean. This procedure is iterated until the uncertainty estimate is self-consistent with the mean parameter. We achieve linear scaling by avoiding to store the covariance explicitly at any time. Instead we draw samples from the approximating distribution relying on an implicit representation and numerical schemes to approximately solve linear equations. Those samples are used to approximate the KL-divergence and its gradient. The usage of natural gradient descent allows for rapid convergence. Formulating the Bayesian model in standardized coordinates makes MGVI applicable to any inference problem with continuous parameters. 


## Citing MGVI.jl

When using MGVI.jl for research, teaching or similar, please cite this publication: [Metric Gaussian Variational Inference](https://arxiv.org/abs/1901.11033)

```
@article{knollmüller2020metric,
         title={Metric Gaussian Variational Inference},
         author={Jakob Knollmüller and Torsten A. Enßlin},
         year={2020},
         eprint={1901.11033},
         archivePrefix={arXiv},
         primaryClass={stat.ML}
}
```
