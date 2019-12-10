using Pkg
Pkg.add("Conda")
Pkg.add("PyCall")
Pkg.add("Logging")
using Conda, PyCall, Logging
try
	Conda.add("numpy")
	Conda.add("scipy")
	Conda.add("cython")
	Conda.add("pip")
	run(`$(PyCall.pyprogramname) -m pip install --user ttpy`)
	@info "Python TT Toolbox installed"
catch;
	@warn "Python TT Toolbox installation failed; the TTPy submodule will not be usable"
end