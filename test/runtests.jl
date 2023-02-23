using Humpty
using Test
using ChainRules, ChainRulesCore

@testset "Humpty.jl" begin
    @testest "$file" for file in ("batch.jl", "execution.jl")
        @include(file)
    end 
end
