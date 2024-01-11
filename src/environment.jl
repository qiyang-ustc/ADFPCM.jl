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
    λ, cl, info = eigsolve(x -> Cmap(x, Tu, Td), Cl, 1, :LM)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], cl[1]
end

function Eenv(Tu, Td, M, Tl)
    λ, al, info = eigsolve(x -> Emap(x, Tu, Td, M), Tl, 1, :LM)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], al[1]
end

function getPL(rt::Runtime, ::FPCM)
    @unpack Tu, Td, Cul, Cld = rt
    λ, Cl = Cenv(Tu, Td, Cul*Cld)
    U, S, V = svd(Array{ComplexF32}(Cl))

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

function leftmove(rt::Runtime, alg::FPCM)
    @unpack M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr = rt
    Cul, Cld, Pl⁺, Pl⁻ = Zygote.@ignore getPL(rt, alg)

    _, Cul = Cenv(Tu, Pl⁻, Cul)
    _, Cld = Cenv(Pl⁺, Td, Cld)
    _, Tl = Eenv(Pl⁺, Pl⁻, M, Tl)

    return Runtime(M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr)
end

function getPL(rt::Runtime, ::CTMRG)
    @unpack M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr = rt
    χ,D = size(Tu)[[1,2]]

    Cu1 = ein"(((abc,cd),def),begh),(((jkl,lm),mna),nhik)->fgij"(Tu,Cul,Tl,M,Tr,Cru,Tu,M)
    Cd1 = ein"(((fed,dc),cba),gebh),(((anm,ml),lkj),ihnk)->fgij"(Tl,Cld,Td,M,Td,Cdr,Tr,M)
    # Cu1 = ein"((jk,kih),hge),ef->fgij"(Cru,Tu,Tu,Cul)
    # Cd1 = ein"((fe,egh),hik),kj->fgij"(Cld,Td,Td,Cdr)

    U, S, V = svd(reshape(ein"fgij,fgkl->klij"(Cu1,Cd1), χ*D,χ*D))

    U = reshape(U[:,1:χ], D, χ, χ)
    V = reshape(V[:,1:χ], D, χ, χ)
    S = S[1:χ]

    sqrtS = sqrt.(S)
    sqrtS⁺ = 1.0 ./sqrtS .* (sqrtS.>1E-7)

    Pl⁺ = ein"(fgij,ijk),kl->lgf"(Cu1, V, Diagonal(sqrtS⁺))
    Pl⁻ = ein"(fgij,ijk),kl->fgl"(Cd1, conj(U), Diagonal(sqrtS⁺))
    
    return Pl⁺, Pl⁻
end

function leftmove(rt::Runtime, alg::CTMRG)
    @unpack M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr = rt
    Pl⁺, Pl⁻ = getPL(rt, alg)

    Cul = Cmap(Cul, Tu, Pl⁻)
    Cld = Cmap(Cld, Pl⁺, Td)
    Tl = Emap(Tl, Pl⁺, Pl⁻, M)
    normalize!(Cul)
    normalize!(Cld)
    normalize!(Tl)

    return Runtime(M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr)
end
