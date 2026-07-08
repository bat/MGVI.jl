# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

module MGVIDistributionsADExt

using DistributionsAD: TuringDenseMvNormal, TuringDiagMvNormal, TuringScalMvNormal

using LinearAlgebra: Cholesky, Diagonal, UpperTriangular

import MGVI: flat_params, euclidean_coords
using MGVI: sqrt2


# DistributionsAD swaps MvNormal for its Turing* types under AD tracing.
# All methods here must match the variance-based parametrization,
# resp. the values, of their MvNormal counterparts in MGVI.

function flat_params(d::TuringDenseMvNormal)
    μ = d.m
    Σ = d.C.L*d.C.U
    vcat(flat_params(μ), flat_params(UpperTriangular(Σ)))
end

flat_params(d::TuringDiagMvNormal) = vcat(d.m, d.σ .^ 2)

flat_params(d::TuringScalMvNormal) = vcat(d.m, d.σ^2)


euclidean_coords(d::TuringDiagMvNormal) = vcat(d.m ./ d.σ, sqrt2 .* log.(d.σ))

euclidean_coords(d::TuringScalMvNormal) =
    vcat(d.m ./ d.σ, sqrt(2*length(d.m)) * log(d.σ))

# MvNormal with diagonal covariance may get swapped for a dense-Cholesky
# TuringDenseMvNormal under AD tracing:
function euclidean_coords(d::TuringDenseMvNormal{<:AbstractVector,<:Cholesky{<:Real,<:Diagonal}})
    σ = d.C.factors.diag
    vcat(d.m ./ σ, sqrt2 .* log.(σ))
end

end # module MGVIDistributionsADExt
