# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

"""
    MGVI.fisher_information(distribution::Distributions.Distribution)

Get the Fisher information of `distribution` as a positive semi-definite
matrix-shaped operator `ℐ` that supports
`MatrixShapedOperators.rowgram_factor` (`ℐ == F * F'`) - in general a
row-Gram operator that stores only its (generalized, possibly
rectangular) Cholesky factor.

Specializations for custom distribution types must return such an
operator, or a plain hermitian `AbstractMatrix` (which MGVI wraps). Up
to MGVI v0.5 this API used `LinearMap`s instead. For a diagonal Fisher
information `d` use `rowgram_operator(diagonal_operator(sqrt.(d)))` -
like there, the factor should be constructed without value-dependent
branches, so that the result stays compatible with program tracing and
compilation.
"""
function fisher_information end

# The Cholesky factor of a diagonal Fisher information:
_diag_fisher(d::AbstractVector) = rowgram_operator(diagonal_operator(sqrt.(d)))

function fisher_information(dist::Normal)
    _, σval = params(dist)
    inv_σ = inv(σval)
    inv_σ_2 = inv_σ * inv_σ
    _diag_fisher(_svector((inv_σ_2, 2*inv_σ_2)))
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

    meanpart = rowgram_operator(asoperator(cholesky(PositiveFactorizations.Positive, invσ).L))
    covpart_op = rowgram_operator(asoperator(cholesky(PositiveFactorizations.Positive, Symmetric(covpart)).L))
    blockdiag_operator(meanpart, covpart_op)
end

# Fisher information w.r.t. the variances (flat_params of PDiagMat), not
# the standard deviations. Elementwise inv keeps this free of scalar
# branches (compare inv(::Diagonal)), for AD- and tracing-compatibility:
function fisher_information(dist::MvNormal{<:Real,<:PDiagMat})
    Σ⁻¹_diag = inv.(dist.Σ.diag)
    blockdiag_operator(_diag_fisher(Σ⁻¹_diag), _diag_fisher(Σ⁻¹_diag .^ 2 ./ 2))
end

# Fisher information w.r.t. the single variance parameter (flat_params of
# ScalMat):
function fisher_information(dist::MvNormal{<:Real,<:ScalMat})
    v = dist.Σ.value
    n = dist.Σ.dim
    blockdiag_operator(
        _diag_fisher(Fill(inv(v), n)),
        _diag_fisher(_svector((n/(2*v^2),)))
    )
end

function fisher_information(dist::Exponential)
    λ = params(dist)[1]
    inv_l = inv(λ)
    _diag_fisher(_svector((inv_l * inv_l,)))
end

function fisher_information(dist::Poisson)
    λ = params(dist)[1]
    _diag_fisher(_svector((inv(λ),)))
end

# Row-Gram blocks combine into a single row-Gram operator of the
# block-diagonal of their factors, all-diagonal factors collapse into a
# single diagonal:

fisher_information(dist::Product) = blockdiag_operator(fisher_information.(dist.v))

fisher_information(dist::Distributions.ProductDistribution) =
    blockdiag_operator(fisher_information.(vec(dist.dists)))

fisher_information(d::NamedTupleDist) = blockdiag_operator(map(fisher_information, values(d))...)


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
