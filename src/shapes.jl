# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


flat_params(x::Real) = _svector((x,))
flat_params(x::StaticVector{N,T}) where {N,T<:Real} = x
flat_params(x::NTuple{N,T}) where {N,T<:Real} = _svector(x)
flat_params(x::Tuple) where {N,T<:Real} = vcat(map(flat_params, x)...)
flat_params(x::NamedTuple) = flat_params(values(x))
flat_params(x::AbstractVector{<:Real}) = x
flat_params(x::AbstractArray{<:Real}) = vec(x)
flat_params(x::AbstractVector{<:AbstractVector{<:Real}}) = _flatten_vec_of_vec(x)
flat_params(x::AbstractVector{<:NTuple{N,<:Real}}) where N = _flatten_vec_of_vec(x)
flat_params(x::AbstractVector) = _flatten_vec_of_vec(flat_params.(x))
flat_params(A::DiagMatLike{<:Real}) = get_diagonal(A)
flat_params(A::PDMat{<:Real}) = flat_params(UpperTriangular(without_chol(A)))


function flat_params(A::UpperTriangular{<:Real})
    m = A.data
    # ToDo: Improve implementation
    reduce(vcat, [m[1:i, i] for i in 1:size(m, 1)])
end

function ChainRulesCore.rrule(::typeof(flat_params), A::UpperTriangular{<:Real})
    function _unshaped_pullback_uptri(thunked_x)
        x = unthunk(thunked_x)
        # ToDo: Improve implementation
        data = A.data
        ΔA_data = zero(A.data)
        for j in axes(ΔA_data, 1)
            ΔA_data[j, 1:j] = view(x, j*(j-1)÷2+1:j*(j+1)÷2)
        end
        return ChainRulesCore.NoTangent(), ProjectTo(A)(data)
    end

    flat_params(A), _unshaped_pullback_uptri
end


function flat_params(d::Distribution)
    flat_params(params(d))
end

function flat_params(ntd::NamedTupleDist)
    flat_params(values(ntd))
end

function flat_params(dp::Distributions.Product)
    flat_params(params.(dp.v))
end

function flat_params(d::TuringDenseMvNormal)
    μ = d.m
    Σ = d.C.L*d.C.U
    vcat(flat_params(μ), flat_params(Σ))
end
