@testset "Basics" begin
    bb = Batch(([0.0, 2.0], [8.0, 0.0]))

    @test size(bb) == (2,)
    @test bb[1] == [0.0, 2.0]
    @test bb[2] == [8.0, 0.0]

    @test_throws DimensionMismatch Batch{3, Float64}([1.])
end


@testset "Matrix representation of Jacobian" begin
    bb = Batch(([0.0, 2.0], [8.0, 0.0]))
    bmat = convert(Matrix{Float64}, bb)
    @test bmat == [0.0 8.0; 2.0 0.0]

    # TODO: we must test that we give output that agrees with the output format of ForwardDiff.jacobian and FIniteDiff.jacobian
end

@testset "vcat" begin
    @test vcat(Batch((1,2,3)), Batch((4,5))) === Batch((1,2,3,4,5))
    @test vcat(Batch(), Batch()) === Batch()
    @test vcat(Batch(), Batch((1,2))) === Batch((1,2))
    @test vcat(Batch((1,2)), Batch()) === Batch((1,2))

    @test vcat(Batch((1,)), Batch((2,)), Batch((3,))) === Batch((1,2,3))
    @test reduce(vcat, (Batch((1,)), Batch((2,)), Batch((3,)))) === Batch((1,2,3))    

end