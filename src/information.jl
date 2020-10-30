# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

function fisher_information(dist::Normal)
    σval = var(dist)
    inv_σ = inv(σval)
    inv_σ_2 = inv_σ * inv_σ
    inv_σ_4 = inv_σ_2 * inv_σ_2
    res = SVector(inv_σ_2, inv_σ_4/2)
    PDiagMat(res)
end

function fisher_information(dist::MvNormal)
    μval, σval = params(dist)
    invσ = inv(σval)

    μdof, σdof = length(μval), length(σval)
    dof = μdof + σdof
    res = zeros(dof, dof)

    res[1:μdof, 1:μdof] = invσ

    for i in 1:σdof
        for j in 1:σdof
            full_i, full_j = μdof + i, μdof + j

            # translation from flat index of matrix to row/col
            i_row, i_col = (i-1) ÷ μdof + 1, (i-1) % μdof + 1
            j_row, j_col = (j-1) ÷ μdof + 1, (j-1) % μdof + 1

            res[full_i, full_j] = invσ[j_col, i_row] * invσ[i_col, j_row] / 2
        end
    end

    sqrt_res = cholesky(PositiveFactorizations.Positive, res)
    PDMat(res, sqrt_res)
end

function fisher_information(dist::Exponential)
    λ = params(dist)[1]
    inv_l = inv(λ)
    res = SVector(inv_l * inv_l,)
    PDiagMat(res)
end

function _blockdiag_map(A::AbstractVector{PDiagMat{T,SArray{Tuple{M},T,1,M}}}) where {T<:Real,M}
    d = reduce(vcat, map(x -> x.diag, A))
    LinearMap(PDiagMat(d), isposdef=true, ishermitian=true, issymmetric=true)
end

function _blockdiag_map(A::AbstractVector{<:AbstractPDMat})
    res = BlockDiagonal(A)
    LinearMap(res, isposdef=true, ishermitian=true, issymmetric=true)
end

function fisher_information(dist::Product)
    dists = dist.v
    λinformations = fisher_information.(dists)
    _blockdiag_map(λinformations)
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
