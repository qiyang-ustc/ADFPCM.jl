using ADFPCM
using ADFPCM: num_grad
using ADFPCM: Eenv, Cenv, getPL, leftmove
using ChainRulesCore
using CUDA
using KrylovKit
using LinearAlgebra
using TensorKit
using Random
using Test
using Zygote
CUDA.allowscalar(false)

@testset "Zygote with $atype{$dtype}" for atype in [Array], dtype in [Float64, ComplexF64]
    a = TensorMap(atype(randn(dtype, 2,2)), ℂ^2, ℂ^2)
    @test Zygote.gradient(norm, a)[1] ≈ num_grad(norm, a)

    foo1 = x -> norm(TensorMap(atype(Float64[x 2x; 3x 4x]), ℂ^2, ℂ^2))
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1)
end

@testset "linsolve with $atype{$dtype}" for atype in [Array], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    D,d = ℂ^4,ℂ^2
    A = TensorMap(randn, dtype, D*d,D)
    @tensor 工[-1 -2; -3 -4] := A[-1 1; -2] * A'[-4;-3 1]
    λLs, Ls, info = eigsolve(L -> (@tensor y[-1; -4] :=  L[-2; -3] * 工[-1 -2; -3 -4]), TensorMap(randn, dtype, D,D'), 1, :LM)
    λL, L = λLs[1], Ls[1]
    λRs, Rs, info = eigsolve(R -> (@tensor y[-3; -2] :=  工[-1 -2; -3 -4] * R[-4; -1]), TensorMap(randn, dtype, D',D), 1, :LM)
    λR, R = λRs[1], Rs[1]

    dL = TensorMap(randn, dtype, D,D')
    dL -= dot(L,dL) * L
    @test dot(L,dL) ≈ 0 atol = 1e-9
    ξL, info = linsolve(R -> (@tensor y[-3; -2] :=  工[-1 -2; -3 -4] * R[-4; -1]), dL', -λL, 1)
    @test dot(ξL',L) ≈ 0 atol = 1e-9

    dR = TensorMap(randn, dtype, D',D)
    dR -= dot(R, dR) * R
    @test dot(R, dR) ≈ 0 atol = 1e-9
    ξR, info = linsolve(L -> (@tensor y[-1; -4] :=  L[-2; -3] * 工[-1 -2; -3 -4]), dR', -λR, 1)
    @test dot(ξR',R) ≈ 0 atol = 1e-9
end

@testset "loop_einsum mistake with $atype{$dtype}" for atype in [Array], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    D = ℂ^10
    A = TensorMap(randn, dtype, D*D,D)
    B = TensorMap(randn, dtype, D,D)
    function foo(x)
        C = A * x
        D = B * x
        E = norm(C)
        F = norm(D)
        return E/F
    end 
    @test Zygote.gradient(foo, 1)[1] ≈ num_grad(foo, 1) atol = 1e-8
end

@testset "Eenv and Cenv with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    χ, D = ℂ^3, ℂ^2

    Tu = TensorMap(randn, ComplexF64, χ*D',χ)
    Tl = TensorMap(randn, ComplexF64, χ*D',χ)
    Td = TensorMap(randn, ComplexF64, χ*D,χ)
    Tr = TensorMap(randn, ComplexF64, χ*D,χ)
    C = TensorMap(randn, ComplexF64, χ,χ)
    E = TensorMap(randn, ComplexF64, χ*D',χ)
    M = TensorMap(randn, ComplexF64, D*D,D*D)
    S1 = TensorMap(randn, ComplexF64, χ'*D*χ, χ*D'*χ')
    S2 = TensorMap(randn, ComplexF64, χ'*χ, χ*χ')

    function foo1(M)
        _, Tl = Eenv(Tu, Td, M, Tl)
        @tensor A = Tl[1 2; 3] * S1[1 2 3;4 5 6] * Tl[4 5; 6]
        B = norm(Tl)
        return norm(A/B)
    end 
    # @test Zygote.gradient(foo1, M)[1] ≈ num_grad(foo1, M) atol = 1e-8

    function foo2(Tu)
        _, C = Cenv(Tu, Td, C)
        @tensor A = C[1; 2] * S2[1 2; 3 4] * C[3; 4]
        B = norm(C)
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