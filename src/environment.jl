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

function Cenv(Tu, Td, Cl; kwargs...)
    λ, cl, info = eigsolve(x -> Cmap(x, Tu, Td), Cl, 1, :LM; kwargs...)
    # info.converged == 0 && error("eigsolve did not converge")
    return λ[1], cl[1]
end

function Eenv(Tu, Td, M, Tl)
    λ, al, info = eigsolve(x -> Emap(x, Tu, Td, M), Tl, 1, :LM)
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
    λ, cul, info = eigsolve(x -> CTMmap(x, Tu, Tl, Td, Tr, M), Cul, 1, :LM)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], cul[1]
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
    
    # for i in 1:10
    #     Cul, Cld, Cul⁺, Cdl⁺, Pl⁺, Pl⁻ = reorthgonal(Cul, Cld, Cul⁺, Cdl⁺, Pl⁺, Pl⁻)
    # end

    return Cul, Cdl, Pl⁺, Pl⁻
end


function filltwo(Pl⁺, Pl⁻)
    Plm⁺ = reshape(Pl⁺, :, size(Pl⁺,3))
    Plm⁻ = reshape(permutedims(Pl⁻, (3,2,1)), :, size(Pl⁻,1))

    q1,r1 = qr(Plm⁺)
    q2,r2 = qr(Plm⁻)
    q1, q2 = Array(q1),Array(q2)

    # @warn "Finding 0 in diagonals, fill with 1.0"
    filltail = (abs.(diag(r1)).*abs.(diag(r2))) .< 1E-12

    r1,r2 = r1 + Diagonal(filltail*1.0), r2 + Diagonal(filltail*1.0)
    q2[:,filltail] .= q1[:,filltail]
    
    Pl⁺ = reshape(q1*r1, size(Pl⁺,1), size(Pl⁺,2), size(Pl⁺,3))
    Pl⁻ = reshape(q2*r2, size(Pl⁻,1), size(Pl⁻,2), size(Pl⁻,3))
    
    # PI = (q1*r1)'*(q2*r2)
    # PIstability = norm(abs.(PI) - Diagonal(ones(size(PI,1))),1)
    # if PIstability > 1E-5
    #     @show q1'*q2
    #     @show r1'*r2
    #     print(PI)
    #     error("PI is not identity, in filling:$(PIstability)")
    # end
    return Pl⁺, permutedims(Pl⁻,(3,2,1))
end

function reorthgonal(Cul, Cdl, Cul⁺, Cdl⁺, Pl⁺, Pl⁻)
    Pl⁺, Pl⁻ = filltwo(Pl⁺, Pl⁻) # makesure Pl+ Pl- full rank
    λ, Yl = Cenv(Pl⁺, Pl⁻, typeof(Cul)(Diagonal(ones(size(Cul)...))))
    U, S, V = svd(Yl)
    sqrtS = sqrt.(S)
    # @show sqrtS
    sqrtS⁺ = 1.0 ./sqrtS .* (sqrtS.>1E-7)
    Yul = U * Diagonal(sqrtS)
    Ydl = Diagonal(sqrtS) * V'
    Yul⁺ = Diagonal(sqrtS⁺) * U'
    Ydl⁺ = V * Diagonal(sqrtS⁺)
    Pl⁺ = ein"(pl,lkj),ji->pki"(Yul⁺,Pl⁺,Yul)
    Pl⁻ = ein"(ij,jkl),lp->ikp"(Ydl,Pl⁻,Ydl⁺)
    Cul = Cul * Yul
    Cdl = Ydl * Cdl
    Cul⁺ = Yul⁺ * Cul⁺
    Cdl⁺ = Cdl⁺ * Ydl⁺
    return Cul, Cdl, Cul⁺, Cdl⁺, Pl⁺, Pl⁻
end

function leftmove(rt::Runtime, alg::FPCM)
    @unpack M, Cul, Cld,  Cdr, Cru, Tu, Tl, Td, Tr = rt
    Cul, Cld, Pl⁺, Pl⁻ = Zygote.@ignore getPL(rt, alg)

    _, Cul = Cenv(Tu, Pl⁻, Cul)
    _, Cld = Cenv(Pl⁺, Td, Cld)
    _, Tl = Eenv(Pl⁺, Pl⁻, M, Tl)

    # _, _, Pu⁺, Pu⁻ = Zygote.@ignore getPL(cycle(cycle(cycle(rt))), alg)
    # _, _, Pd⁺, Pd⁻ = Zygote.@ignore getPL(cycle(rt), alg)
    # _, Cul = CTMenv(Tu, Tl, Pl⁻, Pu⁺, M, Cul)
    # _, Cld = CTMenv(Tl, Td, Pd⁻, Pl⁺, permutedims(M,(2,3,4,1)), Cld)
    

    return Runtime(M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr)
end

## https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.235148 A1-A4
# function getPL(rt::Runtime, ::CTMRG)
#     @unpack M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr = rt
#     χ,D = size(Tu)[[1,2]]

#     Cu1 = ein"(((abc,cd),def),begh),(((jkl,lm),mna),nhik)->fgij"(Tu,Cul,Tl,M,Tr,Cru,Tu,M)
#     Cd1 = ein"(((fed,dc),cba),gebh),(((anm,ml),lkj),ihnk)->fgij"(Tl,Cld,Td,M,Td,Cdr,Tr,M)
#     # Cu1 = ein"((jk,kih),hge),ef->fgij"(Cru,Tu,Tu,Cul)
#     # Cd1 = ein"((fe,egh),hik),kj->fgij"(Cld,Td,Td,Cdr)

#     U, S, V = svd(reshape(ein"fgij,fgkl->klij"(Cu1,Cd1), χ*D,χ*D))

#     U = reshape(U[:,1:χ], D, χ, χ)
#     V = reshape(V[:,1:χ], D, χ, χ)
#     S = S[1:χ]

#     sqrtS = sqrt.(S)
#     sqrtS⁺ = 1.0 ./sqrtS .* (sqrtS.>1E-7)

#     Pl⁺ = ein"(fgij,ijk),kl->lgf"(Cu1, V, Diagonal(sqrtS⁺))
#     Pl⁻ = ein"(fgij,ijk),kl->fgl"(Cd1, conj(U), Diagonal(sqrtS⁺))
    
#     return Pl⁺, Pl⁻
# end

## https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.235148 A5-A8
function getPL(rt::Runtime, ::CTMRG)
    @unpack M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr = rt
    χ,D = size(Tu)[[1,2]]

    Cu1 = ein"(((abc,cd),def),begh),(((jkl,lm),mna),nhik)->fgji"(Tu,Cul,Tl,M,Tr,Cru,Tu,M)
    Cd1 = ein"(((fed,dc),cba),gebh),(((anm,ml),lkj),ihnk)->jifg"(Tl,Cld,Td,M,Td,Cdr,Tr,M)

    Uu, Su, _  = svd(reshape(Cu1, χ*D,χ*D))
    _,  Sd, Vd = svd(reshape(Cd1, χ*D,χ*D))

    # Uu = Uu[:,1:χ]
    # Su = Su[1:χ]
    # Sd = Sd[1:χ]
    # Vd = Vd[:,1:χ]

    Flu = Uu * Diagonal(sqrt.(Su))
    Fdl = Diagonal(sqrt.(Sd)) * Vd'

    Wl, Sl, Ql = svd(Fdl*Flu)
    Wl = Wl[:,1:χ]
    Ql = Ql[:,1:χ]
    Sl = Sl[1:χ]

    sqrtS = Diagonal(sqrt.(Sl))
    sqrtS⁺ = pinv(sqrtS)

    Pl⁺ = ein"abc->cba"(reshape(Flu * Ql * sqrtS⁺, χ,D,χ))
    Pl⁻ = reshape(ein"ab->ba"(Fdl) * conj(Wl) * sqrtS⁺, χ,D,χ)
    
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
