using ADFPCM
using ADFPCM: Cenv, Eenv, getPL
using TeneT: IU1
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
    χ,D = 8,2

    # M = atype(rand(D,D,D,D))
    sitetype = electronZ2()
    dtype = ComplexF64
    M = randU1(sitetype, atype, dtype, D^2,D^2,D^2,D^2; dir = [-1,-1,1,1])

    qnD = getqrange(sitetype, D)
    qnχ = getqrange(sitetype, χ)
    dimsD = getblockdims(sitetype, D)
    dimsχ = getblockdims(sitetype, χ)
    U1info = [qnD..., qnχ..., dimsD..., dimsχ...]
    params = FPCM(χ=χ,verbose=false, U1info = U1info)
    rt = initialize_runtime(M, params)

    λC, Cl = Cenv(rt.Tu, rt.Td, rt.Cul*rt.Cld)
    Cul, Cld, Pl⁺, Pl⁻ = getPL(rt, params)

    @test Cul*Cld ≈ Cl atol = 1e-6

    # # equals identity
    # @show norm(ein"pki,ikl->pl"(Pl⁺,Pl⁻) - IU1(sitetype, atype, dtype, χ; dir = [1,-1]))
    @test ein"pki,ikl->pl"(Pl⁺,Pl⁻) ≈ IU1(sitetype, atype, dtype, χ; dir = [1,-1]) atol = 1e-4

    # Bring to biorthogonal form
    temp = Array(ein"ji,lkj->lki"(Cul,rt.Tu).tensor ./ ein"kji,lk->lji"(Pl⁺, Cul).tensor)
    @test all(x -> isapprox(x,temp[1], atol = 1e-4), temp)

    temp = Array(ein"ij,jkl->ikl"(Cld,rt.Td).tensor ./ ein"ijk,kl->ijl"(Pl⁻, Cld).tensor)
    @test all(x -> isapprox(x,temp[1], atol = 1e-4), temp)
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
