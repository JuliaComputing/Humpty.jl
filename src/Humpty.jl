module Humpty
using ChainRulesCore
export Batch, batched_frule, basis_batch

include("batch.jl")
include("execution.jl")
include("basis.jl")

end
