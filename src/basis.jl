# Here we define generation of Batchs that form a basis of the input space
# This is trivial for the case where all inputs are scalars,
# but for inputs that are vectors or tuples it is more complex
# and for general structs may not be possible (Similar issues to CR.Project, 
# FiniteDifferences.to_vec and CRTU.rand_tangent, you need to know which parts have a 
# Tangent space. Still we mostly pull that off in those existing cases)

# In particular it is a basis comprised of the orthogonal tangent elements
# for the primal input.
# Subject to domain insights, a reduced basis might be possible
# that would be constructable via merging some of the these elements through +

"""
    basis_batch(primal)

Returns a `Batch` containing elements that form a basis of the tangent space of the primal.
For elements with no tangent space this returns an empty Batch.
"""
function basis_batch end


basis_batch(f) = nfields(f)==0 ? Batch() : error("non-singlton structs not supported")

basis_batch(::N) where N<:Number = Batch((oneunit(N),))
basis_batch(x::Integer) = Batch()  # Nondifferentiable

basis_batch(x::Vector{<:Integer}) = Batch()  # Nondifferentiable
function basis_batch(x::Vector{T}) where T<:AbstractFloat
    return ntuple(length(x)) do ii
        ele=zero(x)
        ele[ii] = oneunit(T)
        return ele
    end |> Batch
end

basis_batch(t::Tuple{}) = Batch()
function basis_batch((head, )::T) where T<:Tuple{Any}
    return Batch(map(Tangent{T}, basis_batch(head).elements))
end

function basis_batch((head, tail...)::T) where T<:Tuple
    # TODO: consider dreopping primal type tag except from out-most layer, to decrease work for compiler
    # TODO: optimize this
    ret = []
    for head_base in basis_batch(head)
        push!(ret, (head_base, d_zero(tail)...))
    end
    for tail_base in basis_batch(tail)
        push!(ret, (d_zero(head), tail_base...))
    end

    return ntuple(length(ret)) do ii
        Tangent{T}(ret[ii]...)
    end |> Batch
end

"""
    d_zero(primal)

Returns the additive identity for the primal.
The additive identity is taken from the tangent space.
For primals that have no tangent space returns `NoTangent()`
"""
function d_zero end

d_zero(x::Number) = zero(x)
d_zero(x::AbstractArray) = zero(x)
d_zero(x::Integer) = NoTangent()
d_zero(f) = nfields(f)==0 ? NoTangent() : error("non-singlton structs not supported")
d_zero(t::T) where T<:Tuple = Tangent{T}(map(d_zero, t)...)
