using ADFPCM
using ADFPCM: cycle, leftmove, logZ
using CUDA
using LinearAlgebra
using Random
using Test
using OMEinsum

@testset "FPCMRuntime with $atype" for atype in [Array]
    χ,D = 3,2

    A = atype(rand(χ,D,χ))
    C = atype(rand(χ,χ))
    E = atype(rand(χ,D,χ))
    M = atype(rand(D,D,D,D))

    rt = FPCMRuntime(M,C,C,C,C,A,A,A,A)
    @test rt isa FPCMRuntime
    @test rt.M ≈ M
    @test rt.Cul ≈ C
    @test rt.Cld ≈ C
    @test rt.Cdr ≈ C
    @test rt.Cru ≈ C
    @test rt.Au ≈ A
    @test rt.Al ≈ A
    @test rt.Ad ≈ A
    @test rt.Ar ≈ A
end

@testset "cycle with $atype" for atype in [Array]
    χ,D = 3,2

    Au = atype(rand(χ,D,χ))
    Ad = atype(rand(χ,D,χ))
    C = atype(rand(χ,χ))
    E = atype(rand(χ,D,χ))
    M = atype(rand(D,D,D,D))

    rt = FPCMRuntime(M,C,C,C,C,Au,Au,Au,Au)
    rt = cycle(rt)

    @test rt.M ≈ permutedims(M,(2,3,4,1))
    @test rt.Cul ≈ C
    @test rt.Cld ≈ C
    @test rt.Cdr ≈ C
    @test rt.Cru ≈ C
    @test rt.Au ≈ Au
    @test rt.Al ≈ Au
    @test rt.Ad ≈ Au
    @test rt.Ar ≈ Au
end

@testset "leftmove with $atype" for atype in [Array]
    χ,D = 3,2

    Au = atype(rand(ComplexF64,χ,D,χ))
    Ad = atype(rand(ComplexF64,χ,D,χ))
    C = atype(rand(ComplexF64,χ,χ))
    E = atype(rand(ComplexF64,χ,D,χ))
    M = atype(rand(ComplexF64,D,D,D,D))

    rt = FPCMRuntime(M,C,C,C,C,Au,Au,Au,Au)
    rt = leftmove(rt)

    @test rt isa FPCMRuntime
end

@testset "logZ" for Ni = [1], Nj = [1], atype = [Array]
    D = 2
    χ = 3

    M = atype(rand(ComplexF64,D,D,D,D))
    C = atype(rand(ComplexF64,χ,χ))
    A = atype(rand(ComplexF64,χ,D,χ))

    rt = FPCMRuntime(M,C,C,C,C,A,A,A,A)
    @test isreal(logZ(rt))
end

@testset "FPCM with $atype" for atype in [Array]
    Random.seed!(42)
    M = zeros(ComplexF64, (2,2,2,2))
    M[2,1,1,1]=1.0
    M[1,2,1,1]=1.0
    M[1,1,2,1]=1.0
    M[1,1,1,2]=1.0
    M = atype(M)
    χ = 16
    params = ADFPCM.Params(χ=χ, tol=1e-10, ifsave=false)
    rt = initialize_runtime(M, params)
    rt = FPCM(rt, params)
    @test logZ(rt) ≈ 0.29152163577 atol = 1e-4
end
