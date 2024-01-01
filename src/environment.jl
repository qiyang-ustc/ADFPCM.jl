"""
tensor order graph: from left to right, top to bottom. 
tensor index order: anti-clockwise
```
a ────┬──── c    a──────┬──────b   
│     b     │    │      │      │                     
├─ d ─┼─ e ─┤    │      c      │                  
│     g     │    │      │      │  
f ────┴──── h    d──────┴──────e   

a ────┬──── c  
│     b     │
├─ d ─┼─ e ─┤
│     f     │
├─ g ─┼─ h ─┤           
│     i     │
j ────┴──── k     
```
"""

Cmap(x, Tu, Td) = ein"(bca,ad),dce->be"(Tu, x, Td)
Emap(x, Tu, Td, M) = ein"((cba,adf),bdge),fgh->ceh"(Tu, x, M, Td)

function Cenv(Tu, Td, Cl)
    λ, cl, info = eigsolve(x -> Cmap(x, Tu, Td), Cl, 1, :LM;tol=1E-9)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], cl[1]
end

function Eenv(Tu, Td, M, Tl)
    λ, al, info = eigsolve(x -> Emap(x, Tu, Td, M), Tl, 1, :LM;tol=1E-9)
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

    λCul, Cul = Cenv(Tu, Pl⁻, Cul)
    λCld, Cld = Cenv(Pl⁺, Td, Cld)

    λTl, nTl = Eenv(Pl⁺, Pl⁻, M, Tl)

    movefiedelity = abs(dot(nTl, Tl)/norm(nTl)/norm(Tl))
    FileIO.open("./log/fidelity.log","a") do fid
        write(fid,"$(movefiedelity)\n")
    end
    Tl = nTl

    convergence_Cul = norm(Cmap(Cul, Tu, Pl⁻) - λCul*Cul)
    convergence_Cld = norm(Cmap(Cld, Pl⁺, Td) - λCld*Cld)
    convergence_Tl = norm(Emap(Tl, Pl⁺, Pl⁻, M) - λTl*Tl)

    FileIO.open("./log/eigenconvergence.log","a") do fid
        write(fid,"$(convergence_Cul),$(convergence_Cld),$(convergence_Tl)\n")
    end

    return FPCMRuntime(M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr)
end