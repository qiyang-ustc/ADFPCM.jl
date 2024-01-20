using ADFPCM
using ADFPCM: cycle, leftmove, logZ
using CUDA
using LinearAlgebra
using Random
using TensorKit
using Test

@testset "Runtime with $atype" for atype in [Array]
    χ, D = ℂ^3, ℂ^2
    M = TensorMap(randn, ComplexF64, D*D,D*D)
    Cul = TensorMap(randn, ComplexF64, χ,χ)
    Cld = TensorMap(randn, ComplexF64, χ,χ)
    Cdr = TensorMap(randn, ComplexF64, χ,χ)
    Cru = TensorMap(randn, ComplexF64, χ,χ)
    Tu = TensorMap(randn, ComplexF64, χ*D',χ)
    Tl = TensorMap(randn, ComplexF64, χ*D',χ)
    Td = TensorMap(randn, ComplexF64, χ*D,χ)
    Tr = TensorMap(randn, ComplexF64, χ*D,χ)
    
    rt = Runtime(M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr)
    @test rt isa Runtime
    @test rt.M ≈ M
    @test rt.Cul ≈ Cul
    @test rt.Cld ≈ Cld
    @test rt.Cdr ≈ Cdr
    @test rt.Cru ≈ Cru
    @test rt.Tu ≈ Tu
    @test rt.Tl ≈ Tl
    @test rt.Td ≈ Td
    @test rt.Tr ≈ Tr
end

@testset "cycle with $atype" for atype in [Array]
    χ, D = ℂ^3, ℂ^2
    M = TensorMap(randn, ComplexF64, D*D,D*D)
    Cul = TensorMap(randn, ComplexF64, χ,χ)
    Cld = TensorMap(randn, ComplexF64, χ,χ)
    Cdr = TensorMap(randn, ComplexF64, χ,χ)
    Cru = TensorMap(randn, ComplexF64, χ,χ)
    Tu = TensorMap(randn, ComplexF64, χ*D',χ)
    Tl = TensorMap(randn, ComplexF64, χ*D',χ)
    Td = TensorMap(randn, ComplexF64, χ*D,χ)
    Tr = TensorMap(randn, ComplexF64, χ*D,χ)
    rt = Runtime(M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr)

    rt = cycle(rt)
    @test rt.M ≈ permute(M, ((3,1),(4,2)))
    @test rt.Cul ≈ Cld
    @test rt.Cld ≈ Cdr
    @test rt.Cdr ≈ Cru
    @test rt.Cru ≈ Cul
    @test rt.Tu ≈ Tl
    @test rt.Tl ≈ Td
    @test rt.Td ≈ Tr
    @test rt.Tr ≈ Tu
end

@testset "leftmove with $atype" for atype in [Array]
    χ, D = ℂ^3, ℂ^2
    M = TensorMap(randn, ComplexF64, D*D,D*D)
    params = FPCM(χ=χ,verbose=false)
    rt = initialize_runtime(M, params)

    rt = leftmove(rt, params)

    @test rt isa Runtime
end

@testset "logZ" for Ni = [1], Nj = [1], atype = [Array]
    χ, D = ℂ^3, ℂ^2
    M = TensorMap(randn, ComplexF64, D*D,D*D)
    params = FPCM(χ=χ,verbose=false)
    rt = initialize_runtime(M, params)

    @test isreal(logZ(rt))
end

@testset "FPCM with $atype" for atype in [Array]
    Random.seed!(42)
    χ, D = ℂ^16, ℂ^2
    M = TensorMap(zeros, ComplexF64, D*D,D*D)
    M[2,1,1,1]=1.0
    M[1,2,1,1]=1.0
    M[1,1,2,1]=1.0
    M[1,1,1,2]=1.0
    params = FPCM(χ=χ, tol=1e-10, ifsave=false, verbose=false)
    rt = initialize_runtime(M, params)
    rt = env(rt, params)
    @test logZ(rt) ≈ 0.29152163577 atol = 1e-4
end
