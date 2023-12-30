"""
    i──────┬──────k
    │      │       
    │      j       
    │      │       
    p──────┴──────l
"""
Cmap(x, Au, Ad) = ein"(kji,ip),pjl->kl"(Au, x, Ad)

"""
    i ────┬──── k 
    │     j     
    ├─ a ─┼─ c  
    │     b     
    p ────┴──── l 
"""
Emap(x, Au, Ad, M) = ein"((kji,iap),jabc),pbl->kcl"(Au, x, M, Ad)

function Cenv(Au, Ad, Cl)
    λ, cl, info = eigsolve(x -> Cmap(x, Au, Ad), Cl, 1, :LM)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], cl[1]
end

function Eenv(Pl⁺, Pl⁻, T, Al)
    λ, al, info = eigsolve(x -> Emap(x, Pl⁺, Pl⁻, T), Al, 1, :LM)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], al[1]
end

function getPL(Au, Ad, Cl)
    λ, Cl = Cenv(Au, Ad, Cl)
    U, S, V = svd(Cl)

    sqrtS = sqrt.(S)
    sqrtS⁺ = 1.0 ./sqrtS .* (sqrtS.>1E-7)
    Cul = U * Diagonal(sqrtS)
    Cdl = Diagonal(sqrtS) * V'

    Cul⁺ = Diagonal(sqrtS⁺) * U'
    Cdl⁺ = V * Diagonal(sqrtS⁺)

    Pl⁺ = ein"(pl,lkj),ji->pki"(Cul⁺,Au,Cul)/sqrt(λ)
    Pl⁻ = ein"(ij,jkl),lp->ikp"(Cdl,Ad,Cdl⁺)/sqrt(λ)
    
    return Cul, Cdl, Pl⁺, Pl⁻
end

function leftmove(rt)
    @unpack M, Cul, Cld, Cdr, Cru, Au, Al, Ad, Ar = rt
    Cul, Cld, Pl⁺, Pl⁻ = getPL(Au, Ad, Cul*Cld)

    _, Cul = Cenv(Au, Pl⁻, Cul)
    _, Cld = Cenv(Pl⁺, Ad, Cld)
    _, Al = Eenv(Pl⁺, Pl⁻, M, Al)

    return FPCMRuntime(M, Cul, Cld, Cdr, Cru, Au, Al, Ad, Ar)
end
