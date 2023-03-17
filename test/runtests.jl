using Humpty
using Test
using ChainRulesCore

@testset "Humpty.jl" begin
    @testset "$file" for file in ("batch.jl", "execution.jl", "basis.jl")
        include(file)
    end 
end
