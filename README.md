# TensorTrains

Example [Julia](https://julialang.org/) code for <https://arxiv.org/abs/1802.09062>. This package provides a basic implementation of tensor trains. The functionality implemented here is used by the [TensorTrainFEM](https://github.com/mbachmayr/TensorTrainFEM.jl) package to provide multilevel tensor train finite element solvers.

To install the package, run `using Pkg; Pkg.add("https://github.com/mbachmayr/TensorTrains.jl.git")`.

The main module has the following submodules:
- `TensorTrains.Solvers`, iterative solvers implemented in Julia (following [this paper](http://dx.doi.org/10.1007/s10208-016-9314-z), which provides a rigorous convergence theory).
- `TensorTrains.Condition`, auxiliary routines for evaluating representation condition numbers of tensor train decompositions (see §4 in <https://arxiv.org/abs/1802.09062>).
- `TensorTrains.TTPy`, Julia wrapper for the implementation of the AMEn solver provided by Sergey Dolgov and Dmitry Savostyanov in the [ttpy](https://github.com/oseledets/ttpy) Python package by Ivan Oseledets; to install this optional dependency from within Julia: 
```julia
using Conda, PyCall
Conda.add("numpy")
Conda.add("scipy")
Conda.add("cython")
Conda.add("pip")
run(`$(PyCall.pyprogramname) -m pip install --user ttpy`)
```
These steps are executed automatically when the `TensorTrains` package is built (if these fail, the package is still usable except for the `TTPy` submodule).