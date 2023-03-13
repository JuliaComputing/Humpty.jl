function batched_frule(ẋ_batch::Batch{0}, f, args...)
    return f(args...), Batch()
end

function batched_frule(ẋs::Batch{N}, primal_args...) where N
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
