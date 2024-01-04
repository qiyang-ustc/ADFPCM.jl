"""
tensor order graph: from left to right, top to bottom. 
tensor index order: anti-clockwise
```
a ────┬──── c    a──────┬──────b   
│     b     │    │      │      │                     
├─ d ─┼─ e ─┤    │      c      │                  
│     g     │    │      │      │  
f ────┴──── h    d──────┴──────e   

┌──a──┬──── b 
c     d     │ 
├─ e ─┼─ f ─┤ 
│     g     h 
i ────┴──j  
```
"""


"""
```
    a──────┬──────b 
    │      │      
    │      c      
    │      │      
    d──────┴──────e 
```
"""
Cmap(x, Tu, Td) = ein"(bca,ad),dce->be"(Tu, x, Td)
function Cenv(Tu, Td, Cl)
    λ, cl, info = eigsolve(x -> Cmap(x, Tu, Td), Cl, 1, :LM;tol=1E-9)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], cl[1]
end

"""
```
    a ────┬──── c  
    │     b     │  
    ├─ d ─┼─ e ─┤  
    │     g     │  
    f ────┴──── h  
```
"""
Emap(x, Tu, Td, M) = ein"((cba,adf),bdge),fgh->ceh"(Tu, x, M, Td)
function Eenv(Tu, Td, M, Tl)
    λ, al, info = eigsolve(x -> Emap(x, Tu, Td, M), Tl, 1, :LM;tol=1E-9)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], al[1]
end

"""
```
    ┌──a──┬──── b 
    c     d     │ 
    ├─ e ─┼─ f ─┤ 
    │     g     h 
    i ────┴──j  
```
"""
CTMmap(x, Tu, Tl, Td, Tr, M) = ein"((((bda,ac),cei),degf),igj),hfb->hj"(Tu, x, Tl, M, Td, Tr)
function CTMenv(Tu, Tl, Td, Tr, M, Cul)
    λ, cul, info = eigsolve(x -> CTMmap(x, Tu, Tl, Td, Tr, M), Cul, 1, :LM;tol=1E-9)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], cul[1]
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
    Cul, Cld, Pl⁺, Pl⁻ = Zygote.@ignore getPL(Tu, Td, Cul*Cld)

    _, Cul = Cenv(Tu, Pl⁻, Cul)
    _, Cld = Cenv(Pl⁺, Td, Cld)
    _, Tl = Eenv(Pl⁺, Pl⁻, M, Tl)

    # _, _, Pu⁺, Pu⁻ = Zygote.@ignore getPL(Tr, Tl, Cru*Cul)
    # _, _, Pd⁺, Pd⁻ = Zygote.@ignore getPL(Tl, Tr, Cld*Cdr)
    # _, Cul = CTMenv(Tu, Tl, Pl⁻, Pu⁺, M, Cul)
    # _, Cld = CTMenv(Tl, Td, Pd⁻, Pl⁺, permutedims(M,(2,3,4,1)), Cld)

    return FPCMRuntime(M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr)
end
