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

C = rand(χ,χ)
A = rand(χ,d,χ)

Clu, Cur, Cdl, Crd, Au, Ad, Al, Ar = C,C,C,C,A,A,A,A

CLmap(Au,Ad) = x-> ein"ijk,pjl,ip->kl"(Au,Ad,x)

function leftorth(Au,Ad,Cl)
    res = eigsolve(CLmap(Au,Ad),Cl)
    return res[1][1], res[2][1]
end

function getPL(Au,Ad)
    λ,Cl = leftorth(Au,Ad,rand(χ,χ))
    U,S,V = svd(Cl)
    # U*Diagonal(S)*V' - Cl = 0

    sqrtS = sqrt.(S)
    sqrtS⁺ = 1.0 ./sqrtS .* (sqrtS.>1E-7)
    Clu = Diagonal(sqrtS) * V'
    Cdl = U * Diagonal(sqrtS)

    Clu⁺ = V * Diagonal(sqrtS⁺)
    Cdl⁺ = Diagonal(sqrtS⁺) * U'

    Pl⁺ = ein"ij,jkl,lp->ikp"(Clu,Au,Clu⁺)/sqrt(λ)
    Pl⁻ = ein"ij,jkl,lp->ikp"(Cdl,Ad,Cdl⁺)/sqrt(λ)
    
    # equals identity? 
    # ein"ikp,ikl->pl"(Pl⁺,Pl⁺)

    return Pl⁺, Pl⁻
end

