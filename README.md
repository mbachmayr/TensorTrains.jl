# TensorTrains

Example [Julia](https://julialang.org/) code for <https://arxiv.org/abs/1802.09062>. This package provides a basic implementation of tensor trains. The functionality implemented here is used by the [TensorTrainFEM](https://github.com/mbachmayr/TensorTrainFEM.jl) package to provide multilevel tensor train finite element solvers.

Includes a basic implementation of the tensor train format, assembly of the preconditioned low-rank representations developed in the paper, generic iterative solvers (including a wrapper for AMEn as implemented in [ttpy](https://github.com/oseledets/ttpy)) and some routines for running tests.

To install the package, run `using Pkg; Pkg.add("https://github.com/mbachmayr/TensorTrains.jl.git")`.

The main module has the following submodules:
- `TensorTrains.Solvers`, iterative solvers implemented in Julia (following [this paper](http://dx.doi.org/10.1007/s10208-016-9314-z) providing a rigorous convergence theory).
- `TensorTrains.Condition`, auxiliary routines for evaluating representation condition numbers of tensor train decompositions (see §4 in <https://arxiv.org/abs/1802.09062>).
- `TensorTrains.TTPy`, Julia wrapper for the AMEn implementation provided by Sergey Dolgov and Dmitry Savostyanov in the [ttpy](https://github.com/oseledets/ttpy) Python package by Ivan Oseledets; to install this optional dependency from within Julia: 
```julia
using Conda, PyCall
Conda.add("numpy")
Conda.add("scipy")
Conda.add("cython")
Conda.add("pip")
run(`$(PyCall.pyprogramname) -m pip install --user ttpy`)
```
