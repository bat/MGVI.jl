# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

function fisher_information(dist::Normal)
    _, σval = params(dist)
    inv_σ = inv(σval)
    inv_σ_2 = inv_σ * inv_σ
    inv_σ_4 = inv_σ_2 * inv_σ_2
    res = SVector(inv_σ_2, inv_σ_4/2)
    PDLinMapWithChol(Diagonal(res))
end

function fisher_information(dist::MvNormal)
    μval, σval = params(dist)
    invσ = inv(σval)

    μdof, σdof = length(μval), length(σval)

    covpart = zeros(σdof, σdof)
    for i in 1:σdof
        for j in 1:σdof
            # translation from flat index of matrix to row/col
            i_row, i_col = (i-1) ÷ μdof + 1, (i-1) % μdof + 1
            j_row, j_col = (j-1) ÷ μdof + 1, (j-1) % μdof + 1

            covpart[i, j] = invσ[j_col, i_row] * invσ[i_col, j_row] / 2
        end
    end

    sqrt_meanpart = cholesky(PositiveFactorizations.Positive, invσ).L
    meanpart_map = PDLinMapWithChol(invσ, sqrt_meanpart)

    sqrt_covpart = cholesky(PositiveFactorizations.Positive, covpart).L
    covpart_map = PDLinMapWithChol(covpart, sqrt_covpart)

    blockdiag(meanpart_map, covpart_map)
end

function fisher_information(dist::Exponential)
    λ = params(dist)[1]
    inv_l = inv(λ)
    res = SVector(inv_l * inv_l,)
    PDLinMapWithChol(Diagonal(res))
end

function fisher_information(dist::Product)
    dists = dist.v
    λinformations = fisher_information.(dists)
    _blockdiag_v(λinformations)
end

function fisher_information(d::NamedTupleDist)
    dists = values(d)
    λinformations = map(fisher_information, dists)
    blockdiag(λinformations...)
end

_dists_flat_params_getter(dist_generator) = par::Vector -> reduce(vcat, (par |> dist_generator |> unshaped_params |> values))

function fisher_information_and_jac(f::Function, p::Vector;
                                    jacobian_func::Type{JF}) where JF<:AbstractJacobianFunc
    flat_func = _dists_flat_params_getter(f)
    jac = jacobian_func(flat_func)(p)
    fisher_information(f(p)), jac
end

function fisher_information_in_parspace(λ_fisher::LinearMap, jac::LinearMap)
    adjoint(jac) * λ_fisher * jac
end
