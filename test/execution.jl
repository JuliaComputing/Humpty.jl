
@testset "Basic Batching" begin
    @testset "batched_frule: SISO" begin
        psiso(x) = 100x + 1

        ChainRulesCore.frule((_, ẋ), ::typeof(psiso), x) = psiso(x), 100ẋ
        
        @assert frule((NoTangent(), 2.0), psiso, 3.0) == (301.0, 200.0)
        @assert frule((NoTangent(), 5.0), psiso, 3.0) == (301.0, 500.0)


        res = batched_frule(
            Batch((
                Tangent{Tuple{typeof(psiso),Float64}}(NoTangent(), 2.0),
                Tangent{Tuple{typeof(psiso),Float64}}(NoTangent(), 5.0),
            )),
            psiso, 3.0
        )
        @test res == (301.0, Batch((200.0, 500.0)))
    end

    @testset "batched_frule: MIMO" begin
        # This does not optimize out due to https://github.com/JuliaLang/julia/issues/48808
        # so the primal gets called once for each partial in the batch

        pmimo((x,y)) = [100x+10y, 1000y]

        ChainRulesCore.frule((_, (ẋ, ẏ)), ::typeof(pmimo), xy) = pmimo(xy), [100ẋ+10ẏ, 1000ẏ]
        @assert frule((NoTangent(), [1.0, 0.0]), pmimo, [2, 3]) == ([230, 3000], [100, 0])
        @assert frule((NoTangent(), [0.0, 1.0]), pmimo, [2, 3]) == ([230, 3000], [10, 1000])


        primal_res, deriv_batch = batched_frule(
            Batch((
                Tangent{Tuple{typeof(pmimo), Vector{Float64}}}(NoTangent(), [1.0, 0.0]),
                Tangent{Tuple{typeof(pmimo), Vector{Float64}}}(NoTangent(), [0.0, 1.0]),
            )),
            pmimo, [2, 3]
        )
        @test primal_res == [230, 3000]
        @test deriv_batch == Batch(([100.0, 0.0], [10.0, 1000.0]))

        @testset "Matrix Jacobian" begin
            @test convert(Matrix{Float64}, deriv_batch) == [100 10; 0 1000]  # Checked with ForwardDiff.jacobian(pmimo, [2, 3])
        end
    end

    @testset "batched_frule: MISO" begin
        # This optimizes out so that it only called the primal once
        pmiso((x,y)) = 1000y

        ChainRulesCore.frule((_, (ẋ, ẏ)), ::typeof(pmiso), xy) = pmiso(xy), 1000ẏ
        @assert frule((NoTangent(), [1.0, 0.0]), pmiso, [2, 3]) == (3000, 0)
        @assert frule((NoTangent(), [0.0, 1.0]), pmiso, [2, 3]) == (3000, 1000)


        @test batched_frule(
            Batch((
                Tangent{Tuple{typeof(pmiso), Vector{Float64}}}(NoTangent(), [1.0, 0.0]),
                Tangent{Tuple{typeof(pmiso), Vector{Float64}}}(NoTangent(), [0.0, 2.0]),
                Tangent{Tuple{typeof(pmiso), Vector{Float64}}}(NoTangent(), [3.0, 5.0]),
            )),
            pmiso, [2, 3]
        ) == (3000, Batch((0.0, 2000.0, 5000.0)))
    end
end

# Something where we can't determine the relation without doing AD.
struct NonLinear end
@testset "incidence using" begin

    imiso((x,y,z)) = 10.0y + z^2
    ChainRulesCore.frule((_, (ẋ, ẏ, ż)), ::typeof(imiso), xy) = imiso(xy), 10.0ẏ + 2ż
    @assert frule((NoTangent(), [0.0, 0.0, 0.0]), imiso, [2.0, 3.0, 5.0]) == (55.0, 0.0)
    @assert frule((NoTangent(), [1.0, 0.0, 0.0]), imiso, [2.0, 3.0, 5.0]) == (55.0, 0.0)
    @assert frule((NoTangent(), [0.0, 1.0, 0.0]), imiso, [2.0, 3.0, 5.0]) == (55.0, 10.0)
    @assert frule((NoTangent(), [0.0, 0.0, 1.0]), imiso, [2.0, 3.0, 5.0]) == (55.0, 2.0)


    # using the basis.jl code to make writing tests shorter
    ẋs = basis_batch((imiso, [2.0, 3.0, 5.0]))
    @assert length(ẋs) == 3  # only  3 differentiable elements

    direct_incidence = (0, 10.0, NonLinear())
    _, out = batched_frule(ẋs, direct_incidence, imiso, [2.0, 3.0, 5.0])
    # can't use == as the ZeroTangent breaks it (https://github.com/JuliaDiff/ChainRulesCore.jl/issues/607)
    @test iszero(out[1])
    @test out[2] == 10.0
    @test out[3] == 2.0
end
