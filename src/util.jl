# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

_flatten_vec_of_vec(A::AbstractVector{<:AbstractVector{<:Real}}) = reduce(vcat, A)
_flatten_vec_of_vec(A::AbstractVector{SA}) where {N,T<:Real,SA<:StaticVector{N,T}} = copy(reinterpret(T, A))
# ToDo (if ArraysOfArrays gets added to deps): _flatten_vec_of_vec(A::VectorOfSimilarVectors{<:Real}) = vec(flatview(C))
