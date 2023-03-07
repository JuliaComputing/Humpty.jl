function batched_frule(ẋ_batch::Batch{0}, f, args...)
    return f(args...), Batch()
end

# TODO: Consider deleting this and only allowing Tuple?
# `map` code that works for both doesn't optimize well for Tuple
function batched_frule(ẋs::Batch{N}, primal_args...) where N
    ẋ1 = first(ẋs)
    y, ẏ1 = @inline frule(ẋ1, primal_args...)
    T = typeof(ẏ1)
    ẏs = Vector{T}(undef, N)
    ẏs[1] = ẏ1
    for ii in 2:N
        _, ẏs[ii] = @inline frule(ẋs[ii], primal_args...)
    end

    return y, Batch{N}(ẏs)
end

# Important Tuple Case, make sure it stays Tuple
function batched_frule(ẋs::Batch{N, <:Any, <:Tuple}, primal_args...) where N
    ẋ1 = first(ẋs)
    y, ẏ1 = @inline frule(ẋ1, primal_args...)
    
    ẏs = ntuple(N) do ii
        if ii == 1
            ẏ1
        else
            _, ẏii = @inline frule(ẋs[ii], primal_args...)
            ẏii
        end
    end

    return y, Batch{N}(ẏs)
end
