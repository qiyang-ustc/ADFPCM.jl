using ADFPCM
using ADFPCM: num_grad
using ADFPCM: Eenv, Cenv, getPL, leftmove
using ChainRulesCore
using CUDA
using KrylovKit
using LinearAlgebra
using OMEinsum
using Random
using Test
using Zygote
CUDA.allowscalar(false)

@testset "Zygote with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    a = atype(randn(2,2))
    @test Zygote.gradient(norm, a)[1] ≈ num_grad(norm, a)

    foo1 = x -> sum(atype(Float64[x 2x; 3x 4x]))
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1)
end

@testset "linsolve with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    D,d = 2^2,2
    A = atype(rand(dtype, D,d,D))
    工 = ein"asc,bsd -> abcd"(A,conj(A))
    λLs, Ls, info = eigsolve(L -> ein"ab,abcd -> cd"(L,工), atype(rand(dtype, D,D)), 1, :LM)
    λL, L = λLs[1], Ls[1]
    λRs, Rs, info = eigsolve(R -> ein"abcd,cd -> ab"(工,R), atype(rand(dtype, D,D)), 1, :LM)
    λR, R = λRs[1], Rs[1]

    dL = atype(rand(dtype, D,D))
    dL -= ein"ab,ab -> "(conj(L),dL)[] * L
    @test ein"ab,ab -> "(conj(L),dL)[] ≈ 0 atol = 1e-9
    ξL, info = linsolve(R -> ein"abcd,cd -> ab"(工,R), conj(dL), -λL, 1)
    @test ein"ab,ab -> "(ξL,L)[] ≈ 0 atol = 1e-9

    dR = atype(rand(dtype, D,D))
    dR -= ein"ab,ab -> "(conj(R),dR)[] * R
    @test ein"ab,ab -> "(conj(R),dR)[] ≈ 0 atol = 1e-9
    ξR, info = linsolve(L -> ein"abcd,ab -> cd"(工,L), conj(dR), -λR, 1)
    @test ein"ab,ab -> "(ξR,R)[] ≈ 0 atol = 1e-9
end

@testset "loop_einsum mistake with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    D = 10
    A = atype(rand(dtype, D,D,D))
    B = atype(rand(dtype, D,D))
    function foo(x)
        C = A * x
        D = B * x
        E = ein"abc,abc -> "(C,C)[]
        F = ein"ab,ab -> "(D,D)[]
        return norm(E/F)
    end 
    @test Zygote.gradient(foo, 1)[1] ≈ num_grad(foo, 1) atol = 1e-8
end

@testset "Eenv and Cenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    χ,D = 3,2
    Tu = atype(rand(dtype, χ,D,χ))
    Td = atype(rand(dtype, χ,D,χ))
    Tl = atype(rand(dtype, χ,D,χ))
    C  = atype(rand(dtype, χ,χ))
    S1 = atype(rand(ComplexF64, χ,D,χ,χ,D,χ))
    S2 = atype(rand(ComplexF64, χ,χ,χ,χ))
    M = atype(rand(dtype, D,D,D,D))

    function foo1(M)
        _, Tl = Eenv(Tu, Td, M, Tl)
        A = ein"(abc,abcdef),def -> "(Tl,S1,Tl)[]
        B = ein"abc,abc -> "(Tl,Tl)[]
        return norm(A/B)
    end 
    @test Zygote.gradient(foo1, M)[1] ≈ num_grad(foo1, M) atol = 1e-8

    function foo2(Tu)
        _, C = Cenv(Tu, Td, C)
        A = ein"(ab,abcd),cd -> "(C,S2,C)[]
        B = ein"ab,ab -> "(C,C)[]
        return norm(A/B)
    end 
    @test Zygote.gradient(foo2, Tu)[1] ≈ num_grad(foo2, Tu) atol = 1e-8
end

@testset "svd with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    M = atype(rand(dtype, 5,5))
    function foo(M)
        u, s, v = svd(M)
        return norm(u* Diagonal(s) * v')
    end
    @test Zygote.gradient(foo, M)[1] ≈ num_grad(foo, M) atol = 1e-8
end