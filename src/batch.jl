"""
    Batch{N,D} <: AbstractVector{D}

Represents a batch of tangents that can be propagated together.
`D` must be a valid tangent type, which in short means it must represent an element of a vector space.
Together this batch forms a basis for vector (sub)space.

N is the number of elements in the batch being propagated.
"""
struct Batch{N,D} <: AbstractVector{D}
    elements::Vector{D}
    function Batch{N,D}(elements::Vector{D}) where {N,D}
        N == length(elements) || throw(DimensionMismatch("Size specified as $N, but $(length(elements)) inputs provided."))
        return new{N,D}(elements)
    end
end
Batch(elements::Vector{D}) where D = Batch{length(elements), D}(elements)
Batch{N}(elements::Vector{D}) where {N,D} = Batch{N,D}(elements)
Batch() = Batch([])

Base.size(::Batch{N}) where N = (N,)
Base.getindex(bb::Batch, ii) = bb.elements[ii]

"Convert to a matrix representation of the Jacobian"
function Base.convert(::Type{Matrix{R}}, batch::Batch{N,<:AbstractVector{R}}) where {R<:Real, N}
    N > 0 || return Matrix{R}(0, 0)
    n_out = length(first(batch)) 
    iszero(n_out) && return Matrix{R}(0, 0)

    # we know that for this to be valid all elements must have same length, so can preallocate this
    jac = Matrix{R}(undef, n_out, N)
    for (ii, ele) in zip(axes(jac, 2), batch)
        jac[:, ii] .= ele
    end
    return jac
end