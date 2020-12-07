# MGVInference.jl

[![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://bat.github.io/MGVInference.jl/stable)
[![Documentation for development version](https://img.shields.io/badge/docs-dev-blue.svg)](https://bat.github.io/MGVInference.jl/dev)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Build Status](https://github.com/bat/MGVInference.jl/workflows/CI/badge.svg?branch=master)](https://github.com/bat/MGVInference.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/bat/MGVInference.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/bat/MGVInference.jl)

*This is an implementation of the [Metric Gaussian Variational Inference](https://arxiv.org/abs/1901.11033) (MGVI) algorithm in julia*


MGVI is an iterative method that performs a series of Gaussian approximations to the posterior. We alternate between approximating the covariance with the inverse Fisher information metric evaluated at an intermediate mean estimate and optimizing the KL-divergence for the given covariance with respect to the mean. This procedure is iterated until the uncertainty estimate is self-consistent with the mean parameter. We achieve linear scaling by avoiding to store the covariance explicitly at any time. Instead we draw samples from the approximating distribution relying on an implicit representation and numerical schemes to approximately solve linear equations. Those samples are used to approximate the KL-divergence and its gradient. The usage of natural gradient descent allows for rapid convergence. Formulating the Bayesian model in standardized coordinates makes MGVI applicable to any inference problem with continuous parameters. We demonstrate the high accuracy of MGVI by comparing it to HMC and its fast convergence relative to other established methods in several examples. We investigate real-data applications, as well as synthetic examples of varying size and complexity and up to a million model parameters. 

## Documentation
* [Documentation for stable version](https://bat.github.io/MGVInference.jl/stable)
* [Documentation for development version](https://bat.github.io/MGVInference.jl/dev)
