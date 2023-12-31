using LinearAlgebra
using KrylovKit
using OMEinsum
using CUDA

CLmap(Tu,Td) = x-> ein"ip,kji,pjl->kl"(x,Tu,Td)
CLTmap(Tu,Td,T) = x-> ein"iap,kji,jabc,pbl->kcl"(x,Tu,T,Td)

function leftenv(Tu,Td,Cl)
    res = eigsolve(CLmap(Tu,Td),Cl)
    return res[1][1], res[2][1]
end

function leftenv(Pl⁺, Pl⁻,T,Tl)
    res = eigsolve(CLTmap(Pl⁺, Pl⁻, T),Tl)
    return res[1][1], res[2][1]
end

function getPL(Tu,Td, Cl=rand(χ,χ))
    λ,Cl = leftenv(Tu,Td,Cl)
    U,S,V = svd(Cl)
    # U*Diagonal(S)*V' - Cl = 0

    sqrtS = sqrt.(S)
    sqrtS⁺ = 1.0 ./sqrtS .* (sqrtS.>1E-7)
    Cul = U * Diagonal(sqrtS)
    Cdl = Diagonal(sqrtS) * V'
    # Cul*Cdl - Cl

    Cul⁺ = Diagonal(sqrtS⁺) * U'
    Cdl⁺ = V * Diagonal(sqrtS⁺)
    # Cul*Cul⁺, Cdl*Cdl⁺

    Pl⁺ = ein"ji,lkj,pl->pki"(Cul,Tu,Cul⁺)/sqrt(λ)
    Pl⁻ = ein"ij,jkl,lp->ikp"(Cdl,Td,Cdl⁺)/sqrt(λ)
    
    # equals identity? 
    # ein"pki,ikl->pl"(Pl⁺,Pl⁻)
    # equivalent to: Cul⁺ * ein"ip,kji,pjl->kl"(Cul*Cdl,Tu,Td) * Cdl⁺

    # Bring to biorthogonal form?
    # ein"ji,lkj->lki"(Cul,Tu) ./ ein"kji,lk->lji"(Pl⁺, Cul)
    # ein"ij,jkl->ikl"(Cdl,Td) ./ ein"ijk,kl->ijl"(Pl⁻, Cdl)

    return Cul, Cdl, Pl⁺, Pl⁻
end

function cycle(state)
    Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr, T = state
    state =  Cld, Cdr, Cru, Cul, Tl, Td, Tr, Tu, permutedims(T,(2,3,4,1))
    return state
end

function leftmove(state)
    Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr, T = state
    Cul, Cld, Pl⁺, Pl⁻ = getPL(Tu, Td, Cul*Cld)

    _, Cul = leftenv(Tu,Pl⁻,Cul)
    _, Cld = leftenv(Pl⁺,Td,Cld)
    _, Tl = leftenv(Pl⁺,Pl⁻,T, Tl)

    return Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr, T
end

function logZ(state)
    Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr, T = state
    λT , _ = leftenv(Tu,Td,T,ein"ij,jkl,lp->ikp"(Cul,Tl,Cul))
    λL , _ = leftenv(Tu,Td,Cul*Cld)
    return log(abs(λT/λL))
end

rightmove =  leftmove ∘ cycle ∘ cycle
hvmove = cycle ∘ rightmove ∘ leftmove
rotatemove = cycle ∘ leftmove
cyclemove = rotatemove ∘ rotatemove ∘ rotatemove ∘ rotatemove