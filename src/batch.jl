"""
    Batch{N,D, M} <: AbstractVector{D}

Represents a batch of tangents that can be propagated together.
`D` must be a valid tangent type, which in short means it must represent an element of a vector space.
Together this batch forms a basis for vector (sub)space.

As a general guide all the tangent types stored in the batch will be of type D, but this
does not have to be true as a universal rule, for example they might be `ChainRulesCore.ZeroTangent()`

N is the number of elements in the batch being propagated.
M is the backing storage, and is an implementation detail
"""
struct Batch{N,D,M<:Tuple} <: AbstractVector{D}
    # TODO: do we really need D? or to subtype AbstractVector?
    elements::M
    function Batch{N,D}(elements::M) where {N,D,M}
        N == length(elements) || throw(DimensionMismatch("Size specified as $N, but $(length(elements)) inputs provided."))
        return new{N,D,M}(elements)
    end
end
Batch(elements) = Batch{length(elements), eltype(elements)}(elements)
Batch{N}(elements) where {N} = Batch{N, eltype(elements)}(elements)
Batch() = Batch(tuple())

Base.size(::Batch{N}) where N = (N,)
Base.getindex(bb::Batch, ii) = bb.elements[ii]

"Convert to a matrix representation of the Jacobian"
function Base.convert(::Type{Matrix{R}}, batch::Batch{N,<:AbstractVector{R}}) where {R<:Real, N}
    N > 0 || return Matrix{R}(0, 0)
    n_out = length(first(batch)) 
    iszero(n_out) && return Matrix{R}(0, 0)

    # we know that for this to be valid all elements must have same length, so can preallocate this
    jac = Matrix{R}(undef, n_out, N)
    for (ii, ele) in zip(1:N, batch)
        jac[:, ii] .= ele
    end
    return jac
end


Base.vcat(b::Batch, ::Batch{0}) = b
Base.vcat(::Batch{0}, b::Batch) = b
Base.vcat(b::Batch{0}, ::Batch{0}) = b


function Base.vcat(b1::Batch{N,<:Any,<:Tuple}, b2::Batch{M}) where {N,M}
    return Batch{N+M}(ntuple(N+M) do ii
        if ii <= N
            b1[ii]
        else
            b2[ii-N]
        end
    end
    )
end

# For the important tuple case this constant-folds away
Base.vcat(b1::Batch, bs::Batch...) = reduce(vcat, (b1, bs...))