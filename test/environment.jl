using ADFPCM
using ADFPCM: Cmap, Emap, Cenv, Eenv, getPL
using TensorKit
using CUDA
using LinearAlgebra
using Random
using Test
using OMEinsum

@testset "Cenv and Eenv" begin
    χ, D = ℂ^3, ℂ^2

    Tu = TensorMap(randn, ComplexF64, χ*D',χ)
    Td = TensorMap(randn, ComplexF64, χ*D,χ)
    C = TensorMap(randn, ComplexF64, χ,χ)
    E = TensorMap(randn, ComplexF64, χ*D',χ)
    M = TensorMap(randn, ComplexF64, D*D,D*D)
    λC, C = Cenv(Tu, Td, C)
    λE, E = Eenv(Tu, Td, M, E)

    @test λC * C ≈ Cmap(C, Tu, Td)
    @test λE * E ≈ Emap(E, Tu, Td, M)
end

@testset "fpcm biorthogonal form with $atype" for atype in [Array]
    χ, D = ℂ^2, ℂ^2
    M = TensorMap(randn, ComplexF64, D*D,D*D)
    params = FPCM(χ=χ,verbose=false)
    rt = initialize_runtime(M, params)

    λ, Cl = Cenv(rt.Tu, rt.Td, rt.Cul*rt.Cld)
    Cul, Cdl, Pl⁺, Pl⁻ = getPL(rt, params)

    @test Cul*Cdl ≈ Cl

    # equals identity
    @plansor A[-1;-2] := Pl⁺[-1 1;2] * Pl⁻[2 1;-2]
    @test A ≈ one(A)

    # Bring to biorthogonal form
    @plansor t1[-3 -2;-1] := rt.Tu[-3 -2;1] * Cul[1;-1]
    @plansor t2[-3 -2;-1] := Cul[-3; 1] * Pl⁺[1 -2; -1]
    λ = t1[1]/t2[1]
    @test t1 ≈ λ*t2

    @plansor t1[-1 -2;-3] := Cdl[-1;1] * rt.Td[1 -2;-3]
    @plansor t2[-1 -2;-3] := Pl⁻[-1 -2; 1] * Cdl[1; -3]
    λ = t1[1]/t2[1]
    @test t1 ≈ λ*t2
end

@testset "ctm biorthogonal form with $atype" for atype in [Array]
    χ,D = 3,2

    M = atype(rand(D,D,D,D))
    rt = initialize_runtime(M, ADFPCM.Params(χ=χ))
    Pl⁺, Pl⁻ = getPL(rt)

    # equals identity
    @test Array(ein"pki,ikl->pl"(Pl⁺,Pl⁻)) ≈ I(χ)

    # # Bring to biorthogonal form
    # temp = Array(ein"ji,lkj->lki"(rt.Cul, rt.Tu) ./ ein"kji,lk->lji"(Pl⁺, rt.Cul))
    # @show temp
    # @test all(x -> x ≈ temp[1], temp)

    # temp = Array(ein"ij,jkl->ikl"(Cdl,Td) ./ ein"ijk,kl->ijl"(Pl⁻, Cdl))
    # @test all(x -> x ≈ temp[1], temp)
end

