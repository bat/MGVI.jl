# MGVI.jl

MGVI is an iterative method that performs a series of Gaussian approximations to the posterior. We alternate between approximating the covariance with the inverse Fisher information metric evaluated at an intermediate mean estimate and optimizing the KL-divergence for the given covariance with respect to the mean. This procedure is iterated until the uncertainty estimate is self-consistent with the mean parameter. We achieve linear scaling by avoiding to store the covariance explicitly at any time. Instead we draw samples from the approximating distribution relying on an implicit representation and numerical schemes to approximately solve linear equations. Those samples are used to approximate the KL-divergence and its gradient. The usage of natural gradient descent allows for rapid convergence. Formulating the Bayesian model in standardized coordinates makes MGVI applicable to any inference problem with continuous parameters. 

MGVI.jl also implements geoVI (geometric variational inference, see [`geovi_step`](@ref)), which generalizes MGVI: instead of a multivariate normal distribution in standardized coordinates, the posterior is approximated by pulling a standard normal distribution back through a local isometry of the Fisher metric. This captures non-Gaussian posterior features of forward models that are significantly nonlinear at the posterior's scale, at the cost of one nonlinear least-squares solve per sample. MGVI is the special case of geoVI with a linearized isometry.


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

When using geoVI, please also cite [Geometric Variational Inference](https://arxiv.org/abs/2105.10470)

```
@article{frank2021geometric,
         title={Geometric Variational Inference},
         author={Philipp Frank and Reimar Leike and Torsten A. Enßlin},
         journal={Entropy},
         volume={23},
         number={7},
         pages={853},
         year={2021},
         eprint={2105.10470},
         archivePrefix={arXiv},
         primaryClass={stat.ML}
}
```
