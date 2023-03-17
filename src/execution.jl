function batched_frule(ẋ_batch::Batch{0}, f, args...)
    return f(args...), Batch()
end

# Generic all args version
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


function batched_frule(ẋs::Batch{N}, incidence::Tuple, f, x...) where N
    length(incidence) == N || throw(DimensionMismatch("Must be one incidence element per batch element."))
    y = @inline f(x...)  # TODO: check in all cases we care about this actually optimizes out with the work inside the loop
    ẏs = Batch(ntuple(N) do ii
        ẋ = ẋs[ii]
        🥄 = incidence[ii]

        if 🥄 isa Number
            if iszero(🥄)
                ZeroTangent()  # no work at all, "Sir Not-Appearing-In-This-Film"
            else
                🥄  # It is some linear scaling
            end
        else
            # We don't know how to handle this kind of incidence, (It's probably NonLinear())
            _, ẏii = @inline frule(ẋ, f, x...)
            ẏii
        end
    end)
    return y, ẏs
end
