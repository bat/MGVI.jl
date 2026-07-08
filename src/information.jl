# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

"""
    MGVI.fisher_information(distribution::Distributions.Distribution)

Get the fisher information matrix/operator (as a `LinearMap`) for the given
`distribution`.
"""
function fisher_information end


function fisher_information(dist::Normal)
    _, σval = params(dist)
    inv_σ = inv(σval)
    inv_σ_2 = inv_σ * inv_σ
    res = _svector((inv_σ_2, 2*inv_σ_2))
    PDLinMapWithChol(Diagonal(res))
end

function fisher_information(dist::MvNormal)
    μval, Σ = params(dist)
    σval = without_chol(Σ)
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
    meanpart_map = PDLinMapWithChol(invσ, sqrt_meanpart)

    sym_covpart = Symmetric(covpart)
    sqrt_covpart = cholesky(PositiveFactorizations.Positive, sym_covpart).L
    covpart_map = PDLinMapWithChol(sym_covpart, sqrt_covpart)

    blockdiag(meanpart_map, covpart_map)
end

# Fisher information w.r.t. the variances (flat_params of PDiagMat), not
# the standard deviations:
function fisher_information(dist::MvNormal{<:Real,<:PDiagMat})
    Σ⁻¹ = inv(Diagonal(dist.Σ))
    mean_fisher_map = PDLinMapWithChol(Σ⁻¹)
    cov_fisher_map = PDLinMapWithChol(Σ⁻¹^2/2)
    blockdiag(mean_fisher_map, cov_fisher_map)
end

# Fisher information w.r.t. the single variance parameter (flat_params of
# ScalMat):
function fisher_information(dist::MvNormal{<:Real,<:ScalMat})
    v = dist.Σ.value
    n = dist.Σ.dim
    mean_fisher_map = PDLinMapWithChol(Diagonal(Fill(inv(v), n)))
    cov_fisher_map = PDLinMapWithChol(Diagonal(_svector((n/(2*v^2),))))
    blockdiag(mean_fisher_map, cov_fisher_map)
end

function fisher_information(dist::Exponential)
    λ = params(dist)[1]
    inv_l = inv(λ)
    res = _svector((inv_l * inv_l,))
    PDLinMapWithChol(Diagonal(res))
end

function fisher_information(dist::Poisson)
    λ = params(dist)[1]
    res = _svector((inv(λ),))
    PDLinMapWithChol(Diagonal(res))
end

function fisher_information(dist::Product)
    dists = dist.v
    λinformations = fisher_information.(dists)
    _blockdiag(λinformations)
end

function fisher_information(dist::Distributions.ProductDistribution)
    dists = vec(dist.dists)
    λinformations = fisher_information.(dists)
    _blockdiag(λinformations)
end

function fisher_information(d::NamedTupleDist)
    dists = values(d)
    λinformations = map(fisher_information, dists)
    _blockdiag(λinformations)
end


"""
    MGVI.euclidean_coords(d::Distributions.Distribution)

Compute coordinates of the parameters of `d` in which the Fisher
information metric of `d`'s distribution family is the identity.

The Jacobian `C` of `euclidean_coords` with respect to the flat parameters
of `d` (see `MGVI.flat_params`) satisfies `C'C == fisher_information(d)` -
exactly for distribution families with flat Fisher geometry, up to
curvature terms otherwise (e.g. for `Normal`, where the cross terms
between mean and scale vanish only at `μ == 0`).

Used by geoVI (see [`geovi_step`](@ref)) to construct local isometries.
"""
function euclidean_coords end

function euclidean_coords(d::Normal)
    μ, σ = params(d)
    _svector((μ/σ, sqrt2 * log(σ)))
end

function euclidean_coords(d::Exponential)
    θ = params(d)[1]
    _svector((log(θ),))
end

function euclidean_coords(d::Poisson)
    λ = params(d)[1]
    _svector((2*sqrt(λ),))
end

function euclidean_coords(d::MvNormal{<:Real,<:PDiagMat})
    v = d.Σ.diag
    vcat(d.μ ./ sqrt.(v), invsqrt2 .* log.(v))
end

function euclidean_coords(d::MvNormal{<:Real,<:ScalMat})
    v = d.Σ.value
    n = d.Σ.dim
    vcat(d.μ ./ sqrt(v), sqrt(n/2) * log(v))
end

euclidean_coords(d::Product) = _flatten_vec_of_vec(map(euclidean_coords, d.v))

euclidean_coords(d::Distributions.ProductDistribution) =
    _flatten_vec_of_vec(map(euclidean_coords, vec(d.dists)))

euclidean_coords(d::NamedTupleDist) = vcat(map(euclidean_coords, values(d))...)
