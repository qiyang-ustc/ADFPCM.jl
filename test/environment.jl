using ADFPCM
using ADFPCM: Cenv, Eenv, getPL
using CUDA
using LinearAlgebra
using Random
using Test
using OMEinsum

@testset "Cenv and Eenv with $atype" for atype in [Array]
    χ,D = 3,2

    Au = atype(rand(χ,D,χ))
    Ad = atype(rand(χ,D,χ))
    C = atype(rand(χ,χ))
    E = atype(rand(χ,D,χ))
    M = atype(rand(D,D,D,D))
    λC, C = Cenv(Au, Ad, C)
    λE, E = Eenv(Au, Ad, M, E)

    @test λC * C ≈ ein"ip,kji,pjl->kl"(C, Au, Ad)
    @test λE * E ≈ ein"((iap,kji),jabc),pbl->kcl"(E, Au, M, Ad)
end

@testset "biorthogonal form with $atype" for atype in [Array]
    χ,D = 3,2

    Au = atype(rand(χ,D,χ))
    Ad = atype(rand(χ,D,χ))
    C = atype(rand(χ,χ))
    λC, C = Cenv(Au, Ad, C)
    Cul, Cdl, Pl⁺, Pl⁻ = getPL(Au, Ad, C)

    @test Cul*Cdl ≈ C

    # equals identity
    @test Array(ein"pki,ikl->pl"(Pl⁺,Pl⁻)) ≈ I(χ)

    # Bring to biorthogonal form
    temp = Array(ein"ji,lkj->lki"(Cul,Au) ./ ein"kji,lk->lji"(Pl⁺, Cul))
    @test all(x -> x ≈ temp[1], temp)

    temp = Array(ein"ij,jkl->ikl"(Cdl,Ad) ./ ein"ijk,kl->ijl"(Pl⁻, Cdl))
    @test all(x -> x ≈ temp[1], temp)
end
