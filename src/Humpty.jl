module Humpty
using ChainRulesCore
export Batch, batched_frule

include("batch.jl")
include("execution.jl")

end
