@testset "Basics" begin
    bb = Batch(@SVector [[0.0, 2.0], [8.0, 0.0]])

    @test size(bb) == 2
    @test bb[1] == [0.0, 2.0]
    @test bb[2] == [8.0, 0.0]


end


@testset "Matrix representation of Jacobian" begin
    bb = Batch(@SVector [[0.0, 2.0], [8.0, 0.0]])
    bmat = convert(Matrix{Float64}, bb)
    @test bmat == [0.0 8.0; 2.0 0.0]

    # TODO: we must test that we give output that agrees with the output format of ForwardDiff.jacobian and FIniteDiff.jacobian
end