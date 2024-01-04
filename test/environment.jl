using ADFPCM
using ADFPCM: Cenv, Eenv, CTMenv, Cmap, Emap, CTMmap, getPL
using CUDA
using LinearAlgebra
using Random
using Test
using OMEinsum

@testset "Cenv, Eenv and CTMmap with $atype" for atype in [Array]
    χ,D = 3,2

    Tu = atype(rand(χ,D,χ))
    Tl = atype(rand(χ,D,χ))
    Td = atype(rand(χ,D,χ))
    Tr = atype(rand(χ,D,χ))
    C = atype(rand(χ,χ))
    E = atype(rand(χ,D,χ))
    M = atype(rand(D,D,D,D))
    λC, C = Cenv(Tu, Td, C)
    λE, E = Eenv(Tu, Td, M, E)
    λCM, Cul = CTMenv(Tu, Tl, Td, Tr, M, C)

    @test λC * C ≈ Cmap(C, Tu, Td)
    @test λE * E ≈ Emap(E, Tu, Td, M)
    @test λCM * Cul ≈ CTMmap(Cul, Tu, Tl, Td, Tr, M)
end

@testset "biorthogonal form with $atype" for atype in [Array]
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
