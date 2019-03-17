using TensorTrains
export stsolve, inexstsolve, htsolve

function stsolve(L::Int64, D::Int64, residual::Function,
            normest::Float64, invnormest::Float64,
            ɛ::Float64, Θ::Float64 = 0.25,
            ux::Tensor = Vector{Tensor3}(), αx::Float64 = 0.,
            maxiter::Int64 = typemax(Int64))

  μ = 2 / (1/invnormest + normest);
  κ = normest * invnormest;
  ρ = (κ - 1.) / (κ + 1.);

  if length(ux) > 0
    u = ux
  else
    u = Tensor(2^D, L, 0)
  end
  res = norm!(residual(u))

  udiff = (1. - ρ)/(ρ*normest) * res
      # ensures that α is not decreased in first iteration

  if αx > 0.
    α = αx
  else
    α = .5 * μ * res
  end

  println("κ = ", κ, ", ρ = ", ρ, ", μ = ", μ)

  reslist = Float64[]

  i = 0
  while res > ɛ && i < maxiter
    i += 1
    u0 = deepcopy(u)

    r = residual(u)

    println("   res ranks ", ranks(r))
    println("   max ", maximum(ranks(r)))

    res = norm!(r)
    push!(reslist, res)

    if res <= ɛ
      break
    elseif udiff <= .95 * (1-ρ)/(ρ*normest) * res
      α *= Θ
    end

    add!(u, -μ, r)
    softthresh!(u, α)

    add!(u0, -1., u)
    udiff = norm!(u0)

    println(i, ": α = ", α, ", res ", res, ", udiff ", udiff,
          "  max rank ", maximum(ranks(u)))
  end

  return u, α, reslist
end


function inexstsolve(L::Int64, D::Int64, residual::Function,
            normest::Float64, invnormest::Float64,
            ɛ::Float64, Θ::Float64 = 0.25, ν::Float64 = .95)
  μ = 2 / (1/invnormest + normest);
	κ = normest * invnormest;
	ρ = (κ - 1.) / (κ + 1.);

  τ1 = .5*(1. - ρ)/(3. - ρ)
  τ2 = .25*(1. - ρ)

  ρ̂ = (ρ + τ2*(1. + ρ)/(1. - τ2))

  u = Tensor(2^D, L, 0)
  r0, _ = residual(u, 0.)
  res = norm!(r0)

  α = .5 * μ * res

  B = (1. - ρ)*(1. - τ1)*ν / (normest * (1. + τ2) * ρ̂)
  M = min( (1. - τ1)*τ2*B / (μ * (1 + τ1 + normest*B)),
      (1. - τ1)^2 * ν*ρ*τ2 / (μ * (ρ*(1. + τ1)*(1. + τ2) + ν*(1. - τ1)*(1. - ρ))) )

  udiff = (B / ν) * res

  println("κ = ", κ, ", ρ = ", ρ, ", μ = ", μ)

  reslist = Float64[]
  uranklist = Int64[]
  rranklist = Int64[]
  iranklist = Int64[]

  η = τ1 * res
  i = 0

  while true
    i += 1
    u0 = deepcopy(u)

    r, imax = residual(u, η)
    res = norm!(r)

    println("   η = ", η, ": max res rank ",
            maximum(ranks(r)), "  internal max ", imax)

    while η > τ1 * res
      η *= .5
      r, imax = residual(u, η)
      res = norm!(r)

      println("   (1) η = ", η, ": max res rank ",
              maximum(ranks(r)), "  internal max ", imax)
    end

    push!(reslist, res + η)
    push!(uranklist, maximum(ranks(u)))
    push!(rranklist, maximum(ranks(r)))
    push!(iranklist, maximum(imax))

    if res + η <= ɛ
      break
    elseif udiff <= B * res
      α *= Θ
      η = τ1 * res
    end

    add!(u, -μ, r)
    softthresh!(u, α)

    diff = add(1., u0, -1., u)
    udiff = norm!(diff)

    println(i, ": α = ", α, ", res ", res, ", udiff ", udiff,
      "  max rank ", maximum(ranks(u)))

    while η > (τ2 / μ) * udiff && η > M * res
      η *= .5
      r, imax = residual(u, η)
      res = norm!(r)

      println("   (2) η = ", η, ": max res rank ",
              maximum(ranks(r)), "  internal max ", imax)

      add!(u, -μ, r)
      softthresh!(u, α)

      diff = add(1., u0, -1., u)
      udiff = norm!(diff)

      println(i, ": α = ", α, ", res ", res, ", udiff ", udiff,
        "  max rank ", maximum(ranks(u)))
    end

  end
  return u, (reslist, uranklist, rranklist, iranklist)
end


function htsolve(L::Int64, D::Int64, residual::Function,
      normest::Real, invnormest::Real, ɛ0::Real,
      u = Tensor(2^D, L, 0), κP::Real = sqrt(L-1.), δ0::Real = 0.)
  α = .05
  θ = .5
  β = .5

  κ1 = 1/(1 + κP*(1+α))
  κ2 = κP*(1+α)*κ1

  ω = 2 / (1/invnormest + normest)
  ρ = (normest*invnormest - 1.) / (normest*invnormest + 1.)

  J = 0
  while ρ^J * (1. + (ω + β)*J) > θ*κ1
    J += 1
  end

  if δ0 > 0.
    ε = δ0
  else
    ε = invnormest * norm(residual(u, 0.)[1])
  end

  reslist = Vector{Float64}[]
  uranklist = Int64[]
  rranklist = Int64[]
  iranklist = Int64[]

  k = 0
  while ε > ɛ0
    k += 1

    for j = 0:J
      η = ρ^(j+1) * ε

      r, mr = residual(u, η)
      rnorm = norm(r)

      push!(reslist, rnorm + η)
      push!(uranklist, maximum(ranks(u)))
      push!(rranklist, maximum(ranks(r)))
      push!(iranklist, maximum(imax))

      println(k, ", ", j, ": ", rnorm + η, "  (", mr, ")")

      if invnormest*(rnorm+β*η) <= θ*κ1*ε
        break
      end

      add!(u, -ω, r)
      svdtrunc!(u, svd!(u), β*η)

      println("   ", ranks(u)')
    end

    ε *= θ
    svdtrunc!(u, svd!(u), κ2*ε)
  end

  return u, (reslist, uranklist, rranklist, iranklist)
end
