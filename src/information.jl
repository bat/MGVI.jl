# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

function fisher_information(dist::Normal)
    _, σval = params(dist)
    inv_σ = inv(σval)
    inv_σ_2 = inv_σ * inv_σ
    res = SVector(inv_σ_2, 2*inv_σ_2)
    PDLinMapWithChol(Diagonal(res))
end

function fisher_information(dist::MvNormal)
    μval, σval = params(dist)
    invσ = inv(σval)

    μdof = length(μval)
    σdof = μdof*(μdof+1) ÷ 2

    T = promote_type(eltype(μval), eltype(σval))
    covpart = UpperTriangular(fill(T(0), σdof, σdof))
    for my in 1:μdof
        m_flat_base = my*(my-1)÷2
        for mx in 1:my
            m_flat = m_flat_base + mx
            for ny in 1:my
                n_flat_base = ny*(ny-1)÷2
                for nx in 1:min(ny, (m_flat - n_flat_base))
                    n_flat = n_flat_base + nx
                    covpart[n_flat, m_flat] = (invσ[ny, mx] * invσ[my, nx] + invσ[mx, nx] * invσ[my, ny])
                end
            end
        end
    end

    for x in 1:μdof
        xflat = x*(x+1)÷2
        covpart[1:xflat, xflat] ./= 2
        covpart[xflat, xflat:end] ./= 2
    end

    sqrt_meanpart = cholesky(PositiveFactorizations.Positive, invσ).L
    meanpart_map = PDLinMapWithChol(invσ.mat, sqrt_meanpart)

    sym_covpart = Symmetric(covpart)
    sqrt_covpart = cholesky(PositiveFactorizations.Positive, sym_covpart).L
    covpart_map = PDLinMapWithChol(sym_covpart, sqrt_covpart)

    blockdiag(meanpart_map, covpart_map)
end

function fisher_information(dist::Exponential)
    λ = params(dist)[1]
    inv_l = inv(λ)
    res = SVector(inv_l * inv_l,)
    PDLinMapWithChol(Diagonal(res))
end

function fisher_information(dist::Poisson)
    λ = params(dist)[1]
    res = SVector(inv(λ),)
    PDLinMapWithChol(Diagonal(res))
end

function fisher_information(dist::Product)
    dists = dist.v
    λinformations = fisher_information.(dists)
    _blockdiag(λinformations)
end

function fisher_information(d::NamedTupleDist)
    dists = values(d)
    λinformations = map(fisher_information, dists)
    _blockdiag(λinformations)
end

_dists_flat_params_getter(dist_generator) = par::Vector -> (par |> dist_generator |> unshaped_params)

function fisher_information_and_jac(f::Function, p::AbstractVector;
                                    jacobian_func::Type{JF}) where JF<:AbstractJacobianFunc
    flat_func = _dists_flat_params_getter(f)
    jac = jacobian_func(flat_func)(p)
    fisher_information(f(p)), jac
end

function fisher_information_in_parspace(λ_fisher::LinearMap, jac::LinearMap)
    adjoint(jac) * λ_fisher * jac
end
