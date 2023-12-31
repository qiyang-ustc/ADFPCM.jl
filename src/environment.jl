"""
    i──────┬──────k
    │      │       
    │      j       
    │      │       
    p──────┴──────l
"""
Cmap(x, Tu, Td) = ein"(kji,ip),pjl->kl"(Tu, x, Td)

"""
    i ────┬──── k 
    │     j     
    ├─ a ─┼─ c  
    │     b     
    p ────┴──── l 
"""
Emap(x, Tu, Td, M) = ein"((kji,iap),jabc),pbl->kcl"(Tu, x, M, Td)

function Cenv(Tu, Td, Cl)
    λ, cl, info = eigsolve(x -> Cmap(x, Tu, Td), Cl, 1, :LM)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], cl[1]
end

function Eenv(Tu, Td, M, Tl)
    λ, al, info = eigsolve(x -> Emap(x, Tu, Td, M), Tl, 1, :LM)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], al[1]
end

function getPL(Tu, Td, Cl)
    λ, Cl = Cenv(Tu, Td, Cl)
    U, S, V = svd(Cl)

    sqrtS = sqrt.(S)
    sqrtS⁺ = 1.0 ./sqrtS .* (sqrtS.>1E-7)
    Cul = U * Diagonal(sqrtS)
    Cdl = Diagonal(sqrtS) * V'

    Cul⁺ = Diagonal(sqrtS⁺) * U'
    Cdl⁺ = V * Diagonal(sqrtS⁺)

    Pl⁺ = ein"(pl,lkj),ji->pki"(Cul⁺,Tu,Cul)/sqrt(λ)
    Pl⁻ = ein"(ij,jkl),lp->ikp"(Cdl,Td,Cdl⁺)/sqrt(λ)
    
    return Cul, Cdl, Pl⁺, Pl⁻
end

function leftmove(rt)
    @unpack M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr = rt
    Cul, Cld, Pl⁺, Pl⁻ = getPL(Tu, Td, Cul*Cld)

    _, Cul = Cenv(Tu, Pl⁻, Cul)
    _, Cld = Cenv(Pl⁺, Td, Cld)
    _, Tl = Eenv(Pl⁺, Pl⁻, M, Tl)

    return FPCMRuntime(M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr)
end
