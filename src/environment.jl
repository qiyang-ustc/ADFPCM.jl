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

LCmap(x, Tu, Td) = ein"(bca,ad),dce->be"(Tu, x, Td)
RCmap(x, Tu, Td) = ein"(bca,be),dce->ad"(Tu, x, Td)

Emap(x, Tu, Td, M) = ein"((cba,adf),bdge),fgh->ceh"(Tu, x, M, Td)

function Cenv(Tu, Td, Cl;)
    λ, cl, info = eigsolve(x -> LCmap(x, Tu, Td), Cl, 1, :LM)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], cl[1]
end

function Eenv(Tu, Td, M, Tl)
    λ, al, info = eigsolve(x -> Emap(x, Tu, Td, M), Tl, 1, :LM)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], al[1]
end

function getPL(Tu, Td, Cl)
    λ, cl, info = eigsolve(x -> LCmap(x, Tu, Td), Cl, 1, :LM)
    λ = λ[1]
    U, S, V = svd(cl[1])

    sqrtS = sqrt.(S)
    sqrtS⁺ = 1.0 ./sqrtS # .* (sqrtS.>1E-7)

    Cul = U * Diagonal(sqrtS)
    Cdl = Diagonal(sqrtS) * V'

    Cul⁺ = Diagonal(sqrtS⁺) * U'
    Cdl⁺ = V * Diagonal(sqrtS⁺)

    Pl⁺ = ein"(pl,lkj),ji->pki"(Cul⁺,Tu,Cul )/sqrt(λ)
    Pl⁻ = ein"(ij,jkl),lp->ikp"(Cdl ,Td,Cdl⁺)/sqrt(λ)
    
    # @show eigsolve(x -> LCmap(x, Pl⁺, Pl⁻), Cl, 1, :LM)[1][1]
    return Cul, Cdl, Pl⁺, Pl⁻
end

function precondition(Tu,Td)
    χ = size(Tu, 1)
    λ, cl, info = eigsolve(x -> LCmap(x, Tu, Td), rand(ComplexF64,χ,χ), 1, :LM)
    λ, cr, info = eigsolve(x -> RCmap(x, Tu, Td), rand(ComplexF64,χ,χ), 1, :LM)

    l,r = cl[1],cr[1]

    u = l * transpose(r)
    d = transpose(l) * r

    wu, vu = eigen(u)
    wd, vd = eigen(d)

    Pu = ein"(pl,lkj),ji->pki"(inv(vu),Tu,vu)
    Pd = ein"(ij,jkl),lp->ikp"(transpose(vd),Td,inv(transpose(vd)))
    
    return Pu,Pd
end

function leftmove(rt)
    @unpack M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr = rt
    # Tu, Td = precondition(Tu, Td)
    _, Tl = Eenv(Tu, Td, M, Tl)
    
    Cul = Matrix{ComplexF64}(I, size(Cul,1), size(Cul,2))
    Cru = Matrix{ComplexF64}(I, size(Cul,1), size(Cul,2))
    Cdr = Matrix{ComplexF64}(I, size(Cul,1), size(Cul,2))
    Cld = Matrix{ComplexF64}(I, size(Cul,1), size(Cul,2))
    # Cul, Cld, Pl⁺, Pl⁻ = getPL(Tu, Td, Cul*Cld)
    # _, Cul = Cenv(Tu, Pl⁻, Cul)
    # _, Cld = Cenv(Pl⁺, Td, Cld)
    # _, Tl = Eenv(Pl⁺, Pl⁻, M, Tl)

    return FPCMRuntime(M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr)
end


# function getPL(Tu, Td, Cl)
#     continue_flag = true # automatic try to use square precondition
#     χ = size(Tu, 1)
#     Cul, Cdl, Pl⁺, Pl⁻ = similar(Cl), similar(Cl), similar(Tu), similar(Td)

#     while continue_flag
#         λ, Cl = Cenv(Tu, Td, Cl)
#         U, S, V = svd(Cl)
    
#         sqrtS = sqrt.(S)
#         if sqrtS[χ] < 1E-7 # too strong precondition, use square precondition
#             sqrtS = sqrt.(sqrtS) # make precondition weaker
#         else
#             continue_flag = false
#         end

#         sqrtS⁺ = 1.0 ./sqrtS # .* (sqrtS.>1E-7)
    
#         Cul = U * Diagonal(sqrtS)
#         Cdl = Diagonal(sqrtS) * V'
    
#         Cul⁺ = Diagonal(sqrtS⁺) * U'
#         Cdl⁺ = V * Diagonal(sqrtS⁺)
    
#         Pl⁺ = ein"(pl,lkj),ji->pki"(Cul⁺,Tu,Cul)/sqrt(λ)
#         Pl⁻ = ein"(ij,jkl),lp->ikp"(Cdl,Td,Cdl⁺)/sqrt(λ)     
    
#     end
    
#     sqrtS⁺ = 1.0 ./sqrtS # .* (sqrtS.>1E-7)
    
#     if sqrtS[32] < 1E-7
#         @show "!!!!!!!!!!!!\n\n\n",sqrtS[32]
#     end

#     Cul = U * Diagonal(sqrtS)
#     Cdl = Diagonal(sqrtS) * V'

#     Cul⁺ = Diagonal(sqrtS⁺) * U'
#     Cdl⁺ = V * Diagonal(sqrtS⁺)

#     Pl⁺ = ein"(pl,lkj),ji->pki"(Cul⁺,Tu,Cul)/sqrt(λ)
#     Pl⁻ = ein"(ij,jkl),lp->ikp"(Cdl,Td,Cdl⁺)/sqrt(λ)
    
#     return Cul, Cdl, Pl⁺, Pl⁻
# end