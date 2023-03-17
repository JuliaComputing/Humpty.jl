
struct Singleton end

@testset "Structs" begin
    @test basis_batch(nothing) == Batch()
    @test basis_batch(sin) == Batch()
    @test basis_batch(Singleton()) == Batch()
end

@testset "Numbers" begin
    @test basis_batch(201) === Batch()
    @test basis_batch(201.0) === Batch((1.0,))
end

@testset "Numeric Vector" begin
    @test basis_batch([10.0, 20.0, 30.0]) == Batch((
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ))
    @test basis_batch([10, 20, 30]) == Batch()
end

@testset "tuple" begin
    @test basis_batch(tuple()) == Batch()
    @test basis_batch((10.0,)) == Batch((Tangent{Tuple{Float64}}(1.0,),))
    @test basis_batch((10.0, 20.0)) == Batch((
        Tangent{Tuple{Float64,Float64}}(1.0, 0.0),
        Tangent{Tuple{Float64,Float64}}(0.0, 1.0),
    ))

    @test basis_batch((10.0, 20.0, 30.0)) == Batch((
        Tangent{Tuple{Float64, Float64, Float64}}(1.0, 0.0, 0.0),
        Tangent{Tuple{Float64, Float64, Float64}}(0.0, 1.0, 0.0),
        Tangent{Tuple{Float64, Float64, Float64}}(0.0, 0.0, 1.0),
    ))    
end

@testset "mixed tuple" begin
    @test basis_batch((sin, 20.0)) == Batch((
        Tangent{Tuple{typeof(sin),Float64}}(NoTangent(), 1.0),
    ))

    @test basis_batch(((10.0, 20.0),30.0)) == Batch((
        Tangent{Tuple{Tuple{Float64, Float64}, Float64}}(Tangent{Tuple{Float64, Float64}}(1.0, 0.0), 0.0),
        Tangent{Tuple{Tuple{Float64, Float64}, Float64}}(Tangent{Tuple{Float64, Float64}}(0.0, 1.0), 0.0),
        Tangent{Tuple{Tuple{Float64, Float64}, Float64}}(Tangent{Tuple{Float64, Float64}}(0.0, 0.0), 1.0),
    ))

    @test basis_batch((30.0, (10.0, 20.0))) == Batch((
        Tangent{Tuple{Float64, Tuple{Float64, Float64}}}(1.0, Tangent{Tuple{Float64, Float64}}(0.0, 0.0)),
        Tangent{Tuple{Float64, Tuple{Float64, Float64}}}(0.0, Tangent{Tuple{Float64, Float64}}(1.0, 0.0)),
        Tangent{Tuple{Float64, Tuple{Float64, Float64}}}(0.0, Tangent{Tuple{Float64, Float64}}(0.0, 1.0)),
    ))
end