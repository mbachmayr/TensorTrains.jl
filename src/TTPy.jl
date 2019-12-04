# To install the required python packages via julia 
# (run automatically when the package is built):
#
# using Conda, PyCall
# Conda.add("numpy")
# Conda.add("scipy")
# Conda.add("cython")
# Conda.add("pip")
# run(`$(PyCall.pyprogramname) -m pip install --user ttpy`)

using PyCall, Logging
using TensorTrains

import TensorTrains.Tensor, TensorTrains.TensorMatrix

export pyvec, pymat, Tensor, TensorMatrix, amen

tt = PyNULL()
amen_pkg = PyNULL()

pyvec(t::Tensor) = tt.vector.from_list(t)
pymat(t::TensorMatrix) = tt.matrix.from_list(t)

Tensor(x::PyObject) = Tensor(tt.vector.to_list(x))
TensorMatrix(x::PyObject) = TensorMatrix(tt.matrix.to_list(x))

amen(A::TensorMatrix, b::Tensor, x::Tensor, ɛ::Float64,
		    kickrank::Integer = 4, nswp::Integer = 20) =
		  Tensor(amen_pkg.amen_solve(pymat(A), pyvec(b), pyvec(x), ɛ, kickrank, nswp))

function __init__()
	try
		copy!(tt, pyimport("tt"))
		copy!(amen_pkg, pyimport("tt.amen"))
	catch;
	    @warn "import of Python TT Toolbox failed"
	end
end
