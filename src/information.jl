# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

function jac_dλ_dθ(f::Function, θ)
    dists = f(θ)
    λs = par -> vcat((values ∘ unshaped_params ∘ f)(par)...)
    jac = ForwardDiff.jacobian(λs, θ)
    jac
end

function fisher_information(dist::Normal)
    μval, σval = params(dist)
    res = spzeros(Float64, 2, 2)
    res[1, 1] = 1/σval
    res[2, 2] = 1/2 * 1/σval^2
    res
end

function fisher_information(dist::MvNormal)
    μval, σval = params(dist)
    invσ = inv(σval)

    μdof, σdof = length(μval), length(σval)
    dof = μdof + σdof
    res = spzeros(dof, dof)

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

    res
end

function fisher_information(dist::Exponential)
    λ = params(dist)[1]
    res = spzeros(1, 1)
    res[1, 1] = 1/λ^2
    res
end

function fisher_information(dist::Product)
    dists = dist.v
    λinformations = map(fisher_information, dists)
    blockdiag(λinformations...)
end

function λ_fisher_information(f::Function, p)
    dists = values(f(p))
    λinformations = map(fisher_information, dists)
    blockdiag(λinformations...)
end

function model_fisher_information(f::Function, p)
    jac = jac_dλ_dθ(f, p)
    Symmetric(transpose(jac) * λ_fisher_information(f, p) * jac)
end

export model_fisher_information
