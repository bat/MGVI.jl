# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

"""
    MGVIContext(rng::AbstractRNG, ad::AutoDiffOperators.ADSelector)

Specifies the linear operator type, RNG and automatic differentiation backend
to be used by MGVI operations.
"""
struct MGVIContext{GTX<:GenContext,AD<:ADSelector}
    gen::GTX
    ad::AD
end
export MGVIContext

MGVIContext(ad::ADSelector) = MGVIContext(GenContext(), ad)
