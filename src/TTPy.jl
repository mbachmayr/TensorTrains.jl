# To install the required python packages via julia:
#
# using Conda, PyCall
# Conda.add("numpy")
# Conda.add("scipy")
# Conda.add("cython")
# @pyimport pip
# args = String[]
# push!(args, "install")
# push!(args, "--user")
# push!(args, "ttpy")
# pip.main(args)

using PyCall
using TensorTrains

export pyvec, pymat, Tensor, TensorMatrix, amen

pyvec(t::Tensor) = nothing
pymat(t::TensorMatrix) = nothing

Tensor(x::PyObject) = nothing
TensorMatrix(x::PyObject) = nothing

amen(A::TensorMatrix, b::Tensor, x::Tensor, ɛ::Float64,
    kickrank::Integer = 4, nswp::Integer = 20) = nothing

try
tt = pyimport("tt")
amen_pkg = pyimport("tt.amen")

pyvec(t::Tensor) = tt.vector[:from_list](t)
pymat(t::TensorMatrix) = tt.matrix[:from_list](t)

Tensor(x::PyObject) = Tensor(tt.vector[:to_list](x))
TensorMatrix(x::PyObject) = TensorMatrix(tt.matrix[:to_list](x))

amen(A::TensorMatrix, b::Tensor, x::Tensor, ɛ::Float64,
    kickrank::Integer = 4, nswp::Integer = 20) =
  Tensor(amen_pkg.amen_solve(pymat(A), pyvec(b), pyvec(x), ɛ, kickrank, nswp))
catch;
end