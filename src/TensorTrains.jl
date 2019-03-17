#--------------------------------------------------------------
# TensorTrains package
#--------------------------------------------------------------
# Basic functionality of the tensor train format
module TensorTrains

using LinearAlgebra

const Tensor3 = Array{Float64,3}
const Tensor4 = Array{Float64,4}

import Base.reverse, Base.transpose, LinearAlgebra.norm, LinearAlgebra.dot

export Tensor, TensorMatrix, Tensor3, Tensor4,
      dimpermute,
      ttones, tteye, ttdelta, ttrandn, ttdiagm,
      sizes, ranks, decompress, revdecompress, unfold, evaluate,
      scale, scale!, hadamard, add!, add,
      dot, norm!, norm, rightOrth!, leftOrth!,
      svd!, svdtrunc!, softthresh!,
      matvec, matvec!, matmat,
      reverse, transpose, ttkron

# tensor train representation
const Tensor = Vector{Tensor3}

# representations of matrices acting on tensor trains
const TensorMatrix = Vector{Tensor4}

module Condition
# Auxiliary routines for evaluating representation condition numbers
# (or respective bounds for matrix representations)
include("Condition.jl")
end  # module Condition

module Algorithms
# Simple solvers
include("Algorithms.jl")
end  # module Algorithms

module TTPy
# Interface to python TT toolbox
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
include("TTPy.jl")
end  # module TTPy

# construct tensor train with mode sizes n and ranks 0
function Tensor(n::Vector{Int64})
  m = Vector{Tensor3}(undef, length(n))
  m[1] = Tensor3(1, n[1], 0)
  for i = 2:(length(m)-1)
    m[i] = Tensor3(0, n[i], 0)
  end
  m[end] = Tensor3(0, n[end], 1)
  return m
end

# construct uninitialized tensor train with mode sizes n and ranks r
function Tensor(n::Vector{Int64}, r::Vector{Int64})
  m = Vector{Tensor3}(undef, length(n))
  m[1] = Tensor3(1, n[1], r[1])
  for i in 2:(length(m)-1)
    m[i] = Tensor3(r[i-1], n[i], r[i])
  end
  m[end] = Tensor3(r[end], n[end], 1)
  return m
end

# construct tensor train of order d with mode sizes n
function Tensor(n::Int64, d::Int64)
  m = Tensor3[Tensor3(1,n,0), fill(Tensor3(0,n,0), d-2)..., Tensor3(0,n,1)]
  return m
end

# construct uninitialized tensor train with order d, mode sizes n, ranks r
function Tensor(n::Int64, d::Int64, r::Int64)
  m = Vector{Tensor3}(undef, d)
  m[1] = Tensor3(1, n, r)
  for i in 2:(length(m)-1)
    m[i] = Tensor3(r, n, r)
  end
  m[d] = Tensor3(r, n, 1)
  return m
end

function Tensor3(n1::Int64, n2::Int64, n3::Int64)
  return Tensor3(undef, n1,n2,n3)
end

function Tensor4(n1::Int64, n2::Int64, n3::Int64, n4::Int64)
  return Tensor4(undef, n1,n2,n3,n4)
end

function TensorMatrix(n::Vector{Tuple{Int64,Int64}})
  m = Vector{Tensor4}(undef, length(n))
  m[1] = Tensor4(1, n[1]..., 0)
  for i = 2:(length(m)-1)
    m[i] = Tensor4(0, n[i]..., 0)
  end
  m[end] = Tensor4(0, n[end]..., 1)
  return m
end

function TensorMatrix(n::Vector{Tuple{Int64,Int64}}, r::Vector{Int64})
  m = Vector{Tensor4}(undef, length(n))
  m[1] = Tensor4(1, n[1]..., r[1])
  for i in 2:(length(m)-1)
    m[i] = Tensor4(r[i-1], n[i]..., r[i])
  end
  m[end] = Tensor4(r[end], n[end]..., 1)
  return m
end

function TensorMatrix(n::Int64, d::Int64)
  m = Tensor4[Tensor4(1,n,n,0), fill(Tensor4(0,n,n,0), d-2)..., Tensor4(0,n,n,1)]
  return m
end

function TensorMatrix(n::Int64, d::Int64, r::Int64)
  m = Vector{Tensor4}(undef, d)
  m[1] = Tensor4(1, n, n, r)
  for i in 2:(length(m)-1)
    m[i] = Tensor4(r, n, n, r)
  end
  m[d] = Tensor4(r, n, n, 1)
  return m
end

# convert tensor matrix to regular tensor
function Tensor(t::TensorMatrix)
  r = Tensor(length(t))
  for i in eachindex(t)
    r[i] = reshape(t[i], (size(t[i],1),
          size(t[i],2)*size(t[i],3),size(t[i],4)))
  end
  return r
end

# convert tensor to tensor matrix with specified mode sizes
function TensorMatrix(t::Tensor, n::Vector{Tuple{Int64,Int64}})
  r = TensorMatrix(length(t))
  for i in eachindex(t)
    r[i] = reshape(t[i], (size(t[i],1), n[i]..., size(t[i],3)))
  end
  return r
end

# returns vector of mode sizes
function sizes(t::Tensor)
  return [size(t[i],2) for i in eachindex(t)]
end

# returns vector of mode size tuples
function sizes(t::TensorMatrix)
  return [(size(t[i],2),size(t[i],3)) for i in eachindex(t)]
end

# returns row or column mode size vectors
function sizes(t::TensorMatrix, i::Int64)
  if i == 1
    return [ size(t[i],2) for i in eachindex(t)]
  elseif i == 2
    return [ size(t[i],3) for i in eachindex(t)]
  end
end

# substitute for permutedims
function dimpermute(T::Array, perm::Vector)
  d = ndims(T)
  Tp = zeros(size(T)[perm])
  idx = zeros(Int64, d)
  for i in 1:length(T)
    rem = i - 1
    for k = 1:d
      b = size(T,k)
      r = fld(rem, b)
      idx[k] = rem - b*r
      rem = r
    end
    ip = idx[perm[end]]
    for k = (d-1):-1:1
      b = size(T,perm[k])
      ip = ip*b + idx[perm[k]]
    end
    ip += 1
    Tp[ip] = T[i]
  end
  return Tp
end

# tensor with 1 at index E and 0 otherwise
function ttdelta(n::Int64, E::Vector{Int64})
  t = Tensor(2, length(E), 1)
  for i in eachindex(t)
    fill!(t[i], 0.)
    t[i][1,E[i],1] = 1.
  end
  return t
end

# tensor with all entries 1
function ttones(L::Int64)
  t = Tensor(2, L, 1)
  for i in eachindex(t)
    t[i][1,:,1] = [1., 1.]
  end
  return t
end

# tensor with normally distributed random cores
function ttrandn(L::Int64)
  t = Tensor(2, L, 1)
  for i in eachindex(t)
    t[i][1,:,1] = randn(2)
  end
  return t
end

# identity matrix
function tteye(L::Int64)
  t = TensorMatrix(2, L, 1)
  for i in eachindex(t)
    t[i][1,:,:,1] = eye(2)
  end
  return t
end

# convert tensor to diagonal matrix
function ttdiagm(t::Tensor)
  L = length(t)
  T = TensorMatrix([(n,n) for n in sizes(t)], ranks(t))
  for k = 1:L
    for i = 1:size(T[k],1)
      for j = 1:size(T[k],4)
        T[k][i,:,:,j] = diagm(t[k][i,:,j])
      end
    end
  end
  return T
end

# turn tt representation into full array
function decompress(t::Tensor)
  s1 = size(t[1])
  r = reshape(t[1], (s1[1]*s1[2], s1[3]))
  for i = 2:length(t)
    r1 = size(r, 1)
    si = size(t[i])
    r = reshape(r*reshape(t[i], (si[1], si[2]*si[3])), (r1*si[2], si[3]))
  end
  return reshape(r, (sizes(t)...))
end

# turn tt representation into full array with reversed order of indices
function revdecompress(t::Tensor)
  tx = reverse(t)
  return decompress(tx)
end

# assemble tensor train with first and/or last ranks > 1
function unfold(t::Tensor)
  s1 = size(t[1])
  r = reshape(t[1], (s1[1]*s1[2], s1[3]))
  for i = 2:length(t)
    r1 = size(r, 1)
    si = size(t[i])
    r = reshape(r*reshape(t[i], (si[1], si[2]*si[3])), (r1*si[2], si[3]))
  end
  return reshape(r, (s1[1],prod(sizes(t)),size(t[end],3)))
end

# assemble matrix tensor train
function unfold(T::TensorMatrix)
  s = [sizes(T)[i][j] for j = 1:2, i in eachindex(sizes(T))]
  t = Tensor(T)
  L = length(t)
  dim = zeros(Int64,2L);
  dim[1:L] = (2L-1):-2:1;
  dim[(L+1):(2L)] = (2L):-2:2;
  return reshape(dimpermute(
    reshape(decompress(t), (s[:]...)),
    dim), (prod(s[1,:]), prod(s[2,:])))
end

# evaluate one entry of tensor represented by tensor train
function evaluate(t::Tensor, idx)
  r = t[1][:,idx[1],:]
  for i = 2:length(t)
    r *= t[i][:,idx[i],:]
  end
  return r
end

# multiply tensor by scalar (in-place)
function scale!(t::Tensor, α::Float64)
  t[1] .*= α
  return t
end

# multiply tensor by scalar (creating copy)
function scale(t::Tensor, α::Float64)
  s = deepcopy(t)
  s[1] .*= α
  return s
end

# multiply tensor matrix by scalar (in-place)
function scale!(t::TensorMatrix, α::Float64)
  t[1] .*= α
  return t
end

# multiply tensor matrix by scalar (creating copy)
function scale(t::TensorMatrix, α::Float64)
  s = deepcopy(t)
  s[1] .*= α
  return s
end

# entry-wise product of tensor trains
function hadamard(t1::Tensor, t2::Tensor)
  r = Tensor(length(t1))
  for n in eachindex(t1)
    s1 = size(t1[n])
    s2 = size(t2[n])
    r[n] = Tensor3(s1[1]*s2[1], s1[2], s1[3]*s2[3])
    for i = 1:s1[1], j = 1:s2[1], k = 1:s1[3], l = 1:s2[3]
      r[n][i+s1[1]*(j-1),:,k+s1[3]*(l-1)] = t1[n][i,:,k].*t2[n][j,:,l]
    end
  end
  return r
end

# add two tensor trains (result in t)
function add!(t::Tensor, s::Tensor)
  return add!(t, 1., s)
end

# add α*s to t (with result in t)
function add!(t::Tensor, α::Float64, s::Tensor)
  for (i,(τ,σ)) in enumerate(zip(t,s))
    if i == 1
      c = Tensor3(1,size(τ,2),size(τ,3)+size(σ,3))
      c[:,:,1:size(τ,3)] = τ;
      c[:,:,(size(τ,3)+1):size(c,3)] = α*σ
    elseif i == length(t)
      c = zeros(size(τ,1)+size(σ,1),size(τ,2),1)
      c[1:size(τ,1),:,:] = τ;
      c[(size(τ,1)+1):size(c,1),:,:] = σ
    else
      c = zeros(size(τ,1)+size(σ,1),size(τ,2),size(τ,3)+size(σ,3))
      c[1:size(τ,1),:,1:size(τ,3)] = τ;
      c[(size(τ,1)+1):size(c,1),:,(size(τ,3)+1):size(c,3)] = σ
    end
    t[i] = c
  end
  return t
end

# compute αt + βs in tt format (returning new tensor)
function add(α::Float64, t::Tensor, β::Float64, s::Tensor)
  r = Tensor(sizes(t))
  for (i,(τ,σ)) in enumerate(zip(t,s))
    if i == 1
      c = Tensor3(1,size(τ,2),size(τ,3)+size(σ,3))
      c[:,:,1:size(τ,3)] = α*τ;
      c[:,:,(size(τ,3)+1):size(c,3)] = β*σ
    elseif i == length(t)
      c = zeros(size(τ,1)+size(σ,1),size(τ,2),1)
      c[1:size(τ,1),:,:] = τ;
      c[(size(τ,1)+1):size(c,1),:,:] = σ
    else
      c = zeros(size(τ,1)+size(σ,1),size(τ,2),size(τ,3)+size(σ,3))
      c[1:size(τ,1),:,1:size(τ,3)] = τ;
      c[(size(τ,1)+1):size(c,1),:,(size(τ,3)+1):size(c,3)] = σ
    end
    r[i] = c
  end
  return r
end

# add βs to tensor matrix t (with result in t)
function add!(t::TensorMatrix, β::Float64, s::TensorMatrix)
  for (i,(τ,σ)) in enumerate(zip(t,s))
    if i == 1
      c = Tensor4(1,size(τ,2),size(τ,3),size(τ,4)+size(σ,4))
      c[:,:,:,1:size(τ,4)] = τ;
      c[:,:,:,(size(τ,4)+1):size(c,4)] = β*σ
    elseif i == length(t)
      c = zeros(size(τ,1)+size(σ,1),size(τ,2),size(τ,3),1)
      c[1:size(τ,1),:,:,:] = τ;
      c[(size(τ,1)+1):size(c,1),:,:,:] = σ
    else
      c = zeros(size(τ,1)+size(σ,1),size(τ,2),size(τ,3),size(τ,4)+size(σ,4))
      c[1:size(τ,1),:,:,1:size(τ,4)] = τ;
      c[(size(τ,1)+1):size(c,1),:,:,(size(τ,4)+1):size(c,4)] = σ
    end
    t[i] = c
  end
  return t
end

# add tensor matrices t and s (result in t)
function add!(t::TensorMatrix, s::TensorMatrix)
  return add!(t, 1., s)
end

function add(α::Float64, t::TensorMatrix, β::Float64, s::TensorMatrix)
  r = TensorMatrix(sizes(t))
  for (i,(τ,σ)) in enumerate(zip(t,s))
    if i == 1
      c = Tensor4(1,size(τ,2),size(τ,3),size(τ,4)+size(σ,4))
      c[:,:,:,1:size(τ,4)] = α*τ;
      c[:,:,:,(size(τ,4)+1):size(c,4)] = β*σ
    elseif i == length(t)
      c = zeros(size(τ,1)+size(σ,1),size(τ,2),size(τ,3),1)
      c[1:size(τ,1),:,:,:] = τ;
      c[(size(τ,1)+1):size(c,1),:,:,:] = σ
    else
      c = zeros(size(τ,1)+size(σ,1),size(τ,2),size(τ,3),size(τ,4)+size(σ,4))
      c[1:size(τ,1),:,:,1:size(τ,4)] = τ;
      c[(size(τ,1)+1):size(c,1),:,:,(size(τ,4)+1):size(c,4)] = σ
    end
    r[i] = c
  end
  return r
end

# apply tensor matrix to tensor (result in new tensor)
function matvec(M::TensorMatrix, t::Tensor)
  r = Tensor(sizes(M, 1))
  matvec!(M, t, r)
  return r
end

# apply tensor matrix M to tensor t, with result in r
# (which needs to have correct order)
function matvec!(M::TensorMatrix, t::Tensor, r::Tensor)
  for m in 1:length(t)
    s = size(t[m])
    S = size(M[m])
    T = t[m]
    r[m] = Tensor3(s[1]*S[1], S[2], s[3]*S[4])
    for i = 1:s[1], j = 1:S[1], k = 1:s[3], l = 1:S[4]
      r[m][i+s[1]*(j-1),:,k+s[3]*(l-1)] = M[m][j,:,:,l]*T[i,:,k]
    end
  end
  for m in (length(t)+1):length(M)
    r[m] = reshape(M[m], (size(M[m],1), size(M[m],2), size(M[m],4)))
  end
  if size(r[end],2) == 1
    x = r[end-1]; y = r[end]
    z = reshape(reshape(x, (size(x,1)*size(x,2), size(x,3)))*reshape(y, size(y,1)),
          (size(x,1), size(x,2), 1) )
    r[end-1] = z
    pop!(r)
  end
end

# compute representation of S * T
function matmat(S::TensorMatrix, T::TensorMatrix)
  R = TensorMatrix(length(S))
  for m in eachindex(S)
    R[m] = matmat(S[m], T[m])
  end
  return R
end

function matmat(S::Tensor4, T::Tensor4)
  σ = size(S)
  τ = size(T)
  R = Tensor4(σ[1]*τ[1], σ[2], τ[3], σ[4]*τ[4])
  for i = 1:τ[1], j = 1:σ[1], k = 1:τ[4], l = 1:σ[4]
    R[i+τ[1]*(j-1),:,:,k+τ[4]*(l-1)] = S[j,:,:,l] * T[i,:,:,k]
  end
  return R
end

# euclidean inner product of tt tensors
function dot(t::Tensor, s::Tensor)
  sizet = size(t[1])
  sizes = size(s[1])
  G = reshape(t[1], (sizet[1]*sizet[2],sizet[3]))' *
      reshape(s[1], (sizes[1]*sizes[2],sizes[3]))
  for i = 2:length(t)
    Gi = (t[i][:,1,:])' * G * s[i][:,1,:]
    for n = 2:size(t[i], 2)
      Gi += (t[i][:,n,:])' * G * s[i][:,n,:]
    end
    G = Gi
  end
  return G[1,1]
end

# rank vector of tt tensor (omitting first and last, assumed =1)
function ranks(t::Tensor)
  rvec = zeros(Int64, length(t)-1)
  for i = 1:(length(t)-1)
    rvec[i] = size(t[i], 3)
  end
  return rvec
end

# rank vector of tt matrix (omitting first and last, assumed =1)
function ranks(t::TensorMatrix)
  rvec = zeros(Int64, length(t)-1)
  for i = 1:(length(t)-1)
    rvec[i] = size(t[i], 4)
  end
  return rvec
end

# euclidean norm of tt tensor (t not modified)
function norm(t::Tensor)
  s = deepcopy(t)
  return norm!(s)
end

# euclidean norm of tt tensor (t is orthogonalized)
function norm!(t::Tensor)
  leftOrth!(t)
  return vecnorm(t[1])
end

# orthogonalize to the right in core i
function rightOrth!(t::Tensor, i::Int64)
  s = size(t[i])
  Q, R = qr(reshape(t[i],(s[1]*s[2],s[3])))
  t[i] = reshape(Q, (s[1],s[2],size(Q,2)))
  s = size(t[i+1])
  t[i+1] = reshape(R*reshape(t[i+1],(s[1],s[2]*s[3])), (size(R,1),s[2],s[3]))
end

# orthogonalize entire tensor to the right
function rightOrth!(t::Tensor)
  for i = 1:(length(t)-1)
    rightOrth!(t, i)
  end
end

# orthogonalize to the left in core i
function leftOrth!(t::Tensor, i::Int64)
  s = size(t[i])
  Q, R = qr(reshape(t[i],(s[1],s[2]*s[3]))')
  t[i] = reshape(Q', (size(Q,2),s[2],s[3]))
  s = size(t[i-1])
  t[i-1] = reshape(reshape(t[i-1],(s[1]*s[2],s[3]))*R', (s[1],s[2],size(R,1)))
end

# orthogonalize entire tensor to the left
function leftOrth!(t::Tensor)
  for i = length(t):-1:2
    leftOrth!(t, i)
  end
end

# bring tensor into tt-svd form, return list of singular value vectors
# of matricizations
function svd!(t::Tensor)
  if minimum(ranks(t)) > 0
    rightOrth!(t)
    svlist = Array{Vector{Float64}}(undef, length(t)-1)
    for i = length(t):-1:2
      s = size(t[i])
      U, σ, V = svd(reshape(t[i], (s[1],s[2]*s[3])))
      svlist[i-1] = σ
      s = (size(V,2), s[2], s[3])
      t[i] = reshape(V', s)
      scale!(U, σ)
      s = size(t[i-1])
      t[i-1] = reshape(reshape(t[i-1], (s[1]*s[2],s[3]))*U, (s[1],s[2],size(U,2)))
    end
  else
    svlist = fill(Float64[], length(t)-1)
    t .= Tensor(sizes(t))
  end
  return svlist
end

# truncate t given in tt-svd form according to singular values, up to
# euclidean norm error ε
# usage example for t of type Tensor:
#   svdtrunc!(t, svd!(t), 1e-6)
function svdtrunc!(t::Tensor, svlist, ɛ)
  rvec = [length(s) for s in svlist]
  svvec = Vector{Tuple{Int64,Int64,Float64}}()
  for i in eachindex(svlist)
    for (j,σ) in enumerate(svlist[i])
      push!(svvec, (i, j, σ))
    end
  end
  sort!(svvec, lt = (x,y)->(x[3]<y[3] || (x[3]==y[3] && x[2]>y[2])))
  s = 0.
  stopi = length(svvec)
  for (i,τ) in enumerate(svvec)
    if s + τ[3]^2 <= ɛ^2
      s += τ[3]^2
    else
      stopi = i - 1
      break
    end
  end
  if stopi > 0
    for τ in svvec[1:stopi]
      rvec[τ[1]] = τ[2]-1
    end
  end
  t[1] = t[1][:,:,1:rvec[1]]
  for i = 2:(length(t)-1)
    t[i] = t[i][1:rvec[i-1],:,1:rvec[i]]
  end
  t[end] = t[end][1:rvec[end],:,:]
  if minimum(ranks(t)) == 0
    t .= Tensor(sizes(t))
  end
  return t
end

# soft thresholding of singular values with threshold α
function softthresh!(t::Tensor, α::Float64)
  if minimum(ranks(t)) > 0
    rightOrth!(t)
    for i = length(t):-1:2
      s = size(t[i])
      U, σ, V = svd(reshape(t[i], (s[1],s[2]*s[3])))
      σ = σ[σ .> α] - α
      r = length(σ)
      s = (r, s[2], s[3])
      t[i] = reshape(V[:,1:r]', s)
      U = U[:,1:r]
      scale!(U, σ)
      s = size(t[i-1])
      t[i-1] = reshape(reshape(t[i-1], (s[1]*s[2],s[3]))*U, (s[1],s[2],size(U,2)))
    end
    if minimum(ranks(t)) == 0
      t .= Tensor(sizes(t))
    end
  else
    t .= Tensor(sizes(t))
  end
  return t
end

# reverse indices in tensor
function reverse(t::Tensor)
	L = length(t)
	s = Tensor(L)
	for l ∈ 1:L
		s[L-l+1] = permutedims(t[l], (3,2,1))
	end
	return s
end

# reverse indices in tensor matrix
function reverse(T::TensorMatrix)
	L = length(T)
	S = TensorMatrix(L)
	for l ∈ 1:L
		S[L-l+1] = permutedims(T[l], (4,2,3,1))
	end
	return S
end

# transpose of tensor matrix
function transpose(T::TensorMatrix)
	L = length(T)
	S = TensorMatrix(L)
	for l ∈ 1:L
		S[l] = permutedims(T[l], (1,3,2,4))
	end
	return S
end

transpose(t::Tensor4) = permutedims(t, [1,3,2,4])

function ttkron(A::Tensor4, B::Tensor4)
  return reshape(permutedims(reshape(A[:]*B[:]',
      (size(A)...,size(B)...)), [1,5,2,6,3,7,4,8]),
      ((size(A,i)*size(B,i) for i = 1:4)...) )
end

function ttkron(A::TensorMatrix, B::TensorMatrix)
  C = TensorMatrix(length(A))
  for n = 1:length(C)
    C[n] = ttkron(A[n], B[n])
  end
  return C
end

function ttkron(A::Tensor3, B::Tensor3)
  return reshape(permutedims(reshape(A[:]*B[:]',
      (size(A)...,size(B)...)), [1,4,2,5,3,6]),
      ((size(A,i)*size(B,i) for i = 1:3)...) )
end

function ttkron(A::Tensor, B::Tensor)
  C = Tensor(length(A))
  for n = 1:length(C)
    C[n] = ttkron(A[n], B[n])
  end
  return C
end

end # module
