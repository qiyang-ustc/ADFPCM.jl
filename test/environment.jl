using ADFPCM
using ADFPCM: Cenv, Eenv, getPL
using CUDA
using LinearAlgebra
using Random
using Test
using OMEinsum

@testset "Cenv and Eenv with $atype" for atype in [Array]
    χ,D = 3,2

    Tu = atype(rand(χ,D,χ))
    Td = atype(rand(χ,D,χ))
    C = atype(rand(χ,χ))
    E = atype(rand(χ,D,χ))
    M = atype(rand(D,D,D,D))
    λC, C = Cenv(Tu, Td, C)
    λE, E = Eenv(Tu, Td, M, E)

    @test λC * C ≈ ein"ip,kji,pjl->kl"(C, Tu, Td)
    @test λE * E ≈ ein"((iap,kji),jabc),pbl->kcl"(E, Tu, M, Td)
end

@testset "fpcm biorthogonal form with $atype" for atype in [Array]
    χ,D = 3,2

    Tu = atype(rand(χ,D,χ))
    Td = atype(rand(χ,D,χ))
    C = atype(rand(χ,χ))
    λC, C = Cenv(Tu, Td, C)
    Cul, Cdl, Pl⁺, Pl⁻ = getPL(Tu, Td, C)

    @test Cul*Cdl ≈ C

    # equals identity
    @test Array(ein"pki,ikl->pl"(Pl⁺,Pl⁻)) ≈ I(χ)

    # Bring to biorthogonal form
    temp = Array(ein"ji,lkj->lki"(Cul,Tu) ./ ein"kji,lk->lji"(Pl⁺, Cul))
    @test all(x -> x ≈ temp[1], temp)

    temp = Array(ein"ij,jkl->ikl"(Cdl,Td) ./ ein"ijk,kl->ijl"(Pl⁻, Cdl))
    @test all(x -> x ≈ temp[1], temp)
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
