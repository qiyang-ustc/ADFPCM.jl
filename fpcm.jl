using LinearAlgebra
using KrylovKit
using OMEinsum

d = 2
χ = 16

M = zeros((2,2,2,2))
M[2,1,1,1]=1.0
M[1,2,1,1]=1.0
M[2,2,1,1]=1.0
M[1,2,2,2]=1.0
M[2,1,2,2]=1.0
M[1,1,2,2]=1.0
# logZ = 0.3230659669

M = zeros((2,2,2,2))
M[2,1,1,1]=1.0
M[1,2,1,1]=1.0
M[1,1,2,1]=1.0
M[1,1,1,2]=1.0
# logZ = 0.29152163577 # (0.29155024471215657


C = rand(χ,χ)
A = rand(χ,d,χ)

Cul, Cld, Cdr, Cru, Au, Al, Ad, Ar = C,C,C,C,A,A,A,A
state = Cul, Cld, Cdr, Cru, Au, Al, Ad, Ar, M

CLmap(Au,Ad) = x-> ein"ip,kji,pjl->kl"(x,Au,Ad)
CLTmap(Au,Ad,T) = x-> ein"iap,kji,jabc,pbl->kcl"(x,Au,T,Ad)

function leftenv(Au,Ad,Cl)
    res = eigsolve(CLmap(Au,Ad),Cl)
    return res[1][1], res[2][1]
end

function leftenv(Pl⁺, Pl⁻,T,Al)
    res = eigsolve(CLTmap(Pl⁺, Pl⁻, T),Al)
    return res[1][1], res[2][1]
end

function getPL(Au,Ad, Cl=rand(χ,χ))
    λ,Cl = leftenv(Au,Ad,Cl)
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

    Pl⁺ = ein"ji,lkj,pl->pki"(Cul,Au,Cul⁺)/sqrt(λ)
    Pl⁻ = ein"ij,jkl,lp->ikp"(Cdl,Ad,Cdl⁺)/sqrt(λ)
    
    # equals identity? 
    # ein"pki,ikl->pl"(Pl⁺,Pl⁻)
    # equivalent to: Cul⁺ * ein"ip,kji,pjl->kl"(Cul*Cdl,Au,Ad) * Cdl⁺

    # Bring to biorthogonal form?
    # ein"ji,lkj->lki"(Cul,Au) ./ ein"kji,lk->lji"(Pl⁺, Cul)
    # ein"ij,jkl->ikl"(Cdl,Ad) ./ ein"ijk,kl->ijl"(Pl⁻, Cdl)

    return Cul, Cdl, Pl⁺, Pl⁻
end

function cycle(state)
    Cul, Cld, Cdr, Cru, Au, Al, Ad, Ar, T = state
    state =  Cld, Cdr, Cru, Cul, Al, Ad, Ar, Au, permutedims(T,(2,3,4,1))
    return state
end

function leftmove(state)
    Cul, Cld, Cdr, Cru, Au, Al, Ad, Ar, T = state
    Cul, Cld, Pl⁺, Pl⁻ = getPL(Au, Ad, Cul*Cdl)

    _, Cul = leftenv(Au,Pl⁻,Cul)
    _, Cld = leftenv(Pl⁺,Ad,Cld)
    _, Al = leftenv(Pl⁺,Pl⁻,T, Al)

    return Cul, Cld, Cdr, Cru, Au, Al, Ad, Ar, T
end

function logZ(state)
    Cul, Cld, Cdr, Cru, Au, Al, Ad, Ar, T = state
    λT , _ = leftenv(Au,Ad,T,ein"ij,jkl,lp->ikp"(Cul,Al,Cul))
    λL , _ = leftenv(Au,Ad,Cul*Cdl)
    return log(abs(λT/λL))
end

rotatemove = cycle ∘ leftmove
cyclemove = rotatemove ∘ rotatemove ∘ rotatemove ∘ rotatemove

for i = 1:100
    state = cyclemove(state)
    @show logZ(state), logZ(cycle(state)),  logZ(cycle(cycle(state))), logZ((cycle ∘cycle ∘ cycle)(state))
end