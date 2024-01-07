using ADFPCM
using ADFPCM: cycle, leftmove, logZ
using CUDA
using LinearAlgebra
using Random
using TeneT
using Test
using OMEinsum


@testset "initialize_runtime with $symmetry and $stype" for symmetry in [:none, :U1], stype in [TeneT.electronZ2()]
    Random.seed!(42)
    M = rand(ComplexF64, (4,4,4,4))
    M = asSymmetryArray(M, Val(symmetry), stype; dir = [-1,-1,1,1])
    rt = initialize_runtime(M, FPCM(χ=16, U1info = [[0,1], [0,1], [1,1], [8,8]]))
    @test rt isa Runtime
    @test rt.M ≈ M
    @test rt.Cul ≈ rt.Cld ≈ rt.Cdr ≈ rt.Cru 
    @test rt.Tu ≈ rt.Tl ≈ rt.Td ≈ rt.Tr
end

@testset "cycle with $symmetry and $stype" for symmetry in [:none, :U1], stype in [TeneT.electronZ2()]
    Random.seed!(42)
    M = rand(ComplexF64, (4,4,4,4))
    M = asSymmetryArray(M, Val(symmetry), stype; dir = [-1,-1,1,1])
    rt = initialize_runtime(M, FPCM(χ=16, U1info = [[0,1], [0,1], [1,1], [8,8]]))
    rt = cycle(rt)

    @test rt.M ≈ permutedims(M,(2,3,4,1))
    @test rt.Cul ≈ rt.Cld
    @test rt.Cld ≈ rt.Cdr
    @test rt.Cdr ≈ rt.Cru
    @test rt.Cru ≈ rt.Cul
    @test rt.Tu ≈ rt.Td
    @test rt.Tl ≈ rt.Tu
    @test rt.Td ≈ rt.Td
    @test rt.Tr ≈ rt.Tu
end

@testset "leftmove with $symmetry and $stype" for symmetry in [:U1], stype in [TeneT.electronZ2()]
    Random.seed!(42)
    M = rand(ComplexF64, (4,4,4,4))
    M = asSymmetryArray(M, Val(symmetry), stype; dir = [-1,-1,-1,-1])
    alg = FPCM(χ=16, U1info = [[0,1], [0,1], [1,1], [8,8]])
    rt = initialize_runtime(M, alg)

    rt = leftmove(rt, alg)

    @test rt isa Runtime
end

@testset "FPCM with $symmetry and $stype" for symmetry in [:U1], stype in [TeneT.electronZ2()]
    Random.seed!(42)
    M = rand(ComplexF64, (4,4,4,4))
    M = asSymmetryArray(M, Val(symmetry), stype; dir = [-1,-1,1,1])
    alg = FPCM(χ=16, U1info = [[0,1], [0,1], [1,1], [8,8]])
    rt = env(M, alg)
end
