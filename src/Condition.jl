using TensorTrains
export repampvals, repamp, repampbound, repcondbound

function repampvals(t::Tensor, l::Int64)
  if l == 1
    T2 = unfold(t[2:end])
    return 1., norm(t[1][:]),
      maximum(svdvals(reshape(T2, (size(T2,1), size(T2,2)))))
  elseif l == length(t)
    T1 = unfold(t[1:(l-1)])
    return maximum(svdvals(reshape(T1, (size(T1,2), size(T1,3))))),
      norm(t[l][:]), 1.
  else
    T1 = unfold(t[1:(l-1)])
    T2 = unfold(t[(l+1):end])
    return maximum(svdvals(reshape(T1, (size(T1,2), size(T1,3))))),
      norm(t[l][:]),
      maximum(svdvals(reshape(T2, (size(T2,1), size(T2,2)))))
  end
end

# computes the exact value of the sum of ramp_l(t) over l
function repamp(t::Tensor)
  s = 0.;
  for l = 1:length(t)
    a, b, c = repampvals(t, l)
    s += a*b*c
  end
  return s
end

# computes the exact value of ramp_l(t)
function repamp(t::Tensor, l::Integer)
  a, b, c = repampvals(t, l)
  return a*b*c
end


function leftUnfold(T::TensorMatrix, l::Int64, i::Int64)
  s = [sizes(T)[i][j] for j = 1:2, i in eachindex(sizes(T))]
  s = s[:,1:(l-1)]
  t = Tensor(T)[1:(l-1)]
  t[l-1] = reshape(t[l-1][:,:,i], (size(t[l-1],1),size(t[l-1],2),1))
  M = unfold(t)[1,:,1]
  L = l-1
  dim = zeros(Int64,2L)
  dim[1:L] = (2L-1):-2:1
  dim[(L+1):(2L)] = (2L):-2:2
  return reshape(dimpermute(reshape(M, (s[:]...,)), dim),
                (prod(s[1,:]), prod(s[2,:])))
end

function rightUnfold(T::TensorMatrix, l::Int64, i::Int64)
  s = [sizes(T)[i][j] for j = 1:2, i in eachindex(sizes(T))]
  s = s[:,(l+1):length(T)]
  t = Tensor(T)[(l+1):length(T)]
  t[1] = reshape(t[1][i,:,:], (1,size(t[1],2),size(t[1],3)))
  M = unfold(t)[1,:,1]
  L = length(T) - l
  dim = zeros(Int64,2L);
  dim[1:L] = (2L-1):-2:1;
  dim[(L+1):(2L)] = (2L):-2:2;
  return reshape(dimpermute(reshape(M, (s[:]...,)), dim),
            (prod(s[1,:]), prod(s[2,:])))
end

function repampvals(T::TensorMatrix, l::Int64)
  cnorms = [norm(T[l][i,:,:,j]) for i=1:size(T[l],1), j=1:size(T[l],4)]
  if l == 1
    rnorms = [norm(rightUnfold(T, l, i)) for i = 1:size(T[l],4)]
    return [1.], cnorms, rnorms
  elseif l == length(T)
    lnorms = [norm(leftUnfold(T, l, i)) for i = 1:size(T[l],1) ]
    return lnorms, cnorms, [1.]
  else
    lnorms = [norm(leftUnfold(T, l, i)) for i = 1:size(T[l],1) ]
    rnorms = [norm(rightUnfold(T, l, i)) for i = 1:size(T[l],4) ]
    return lnorms, cnorms, rnorms
  end
end

function repampbound(T::TensorMatrix, l::Int64)
  a, b, c = repampvals(T, l)
  return norm(a)*vecnorm(b)*norm(c)
end

function repampbound(T::TensorMatrix)
  return sum(repampbound(T,l) for l = 1:length(T))
end

function repcondbound(T::TensorMatrix)
  return repampbound(T) / minimum(svdvals(unfold(T)))
end
