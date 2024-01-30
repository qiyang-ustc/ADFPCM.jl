"""
tensor index order: anti-clockwise
```
    1───┬───2                         
    │   3                     
    4───┴───5  
```
"""
function Cmap(x::AbstractTensorMap{<:IndexSpace, 1,1}, Tu::AbstractTensorMap{<:IndexSpace, 2,1}, Td::AbstractTensorMap{<:IndexSpace, 2,1})
    return @plansor y[-2; -5] := Tu[-2 3; 1] * x[1; 4] * Td[4 3; -5]
end

"""
```
    1─┬─┬─2                            
    │ 3 4                          
    5─┴─┴─6  
```
"""
function Cmap(x::AbstractTensorMap{<:IndexSpace, 1,1}, Tu::AbstractTensorMap{<:IndexSpace, 2,2}, Td::AbstractTensorMap{<:IndexSpace, 2,2}) 
    return @plansor y[-2; -6] := Tu[-2 3; 4 1] * x[1; 5] * Td[5 3; 4 -6]
end

"""
```
    1 ────┬──── 3
    │     2     
    ├─ 4 ─┼─ 5  
    │     6     
    7 ────┴──── 8
```
"""
function Emap(x::AbstractTensorMap{<:IndexSpace, 2,1}, Tu::AbstractTensorMap{<:IndexSpace, 2,1}, Td::AbstractTensorMap{<:IndexSpace, 2,1}, M::AbstractTensorMap{<:IndexSpace, 2,2}) 
    return @plansor y[-3 -5; -8] := Tu[-3 2; 1] * x[1 4; 7] * M[4 2;6 -5] * Td[7 6; -8] 
end

"""
```
    1───┬─┬───4
    │   2 3     
    ├─5─┘ └─6 
    ├─7─┐ ┌─8 
    │   9 10  
    11──┴─┴───12
```
"""
function Emap(x::AbstractTensorMap{<:IndexSpace, 2,2}, Tu::AbstractTensorMap{<:IndexSpace, 2,2}, Td::AbstractTensorMap{<:IndexSpace, 2,2}, M::AbstractTensorMap{<:IndexSpace, 4,4})
    # how to use @plansor?
    return @tensor y[-4 -8; -6 -12] := Tu[-4 2; 3 1] * x[1 7; 5 11] * M[7 5 2 3; 9 10 -8 -6] * Td[11 9; 10 -12] 
end

function Cenv(Tu, Td, Cl; kwargs...)
    λ, cl, info = eigsolve(x -> Cmap(x, Tu, Td), Cl, 1, :LM; kwargs...)
    info.converged == 0 && error("eigsolve did not converge")
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
    U, S, Vt = svd(Cl)

    sqrtS = sqrt(S)
    Cul = U * sqrtS
    Cdl = sqrtS * Vt

    sqrtS⁺ = inv(sqrtS)
    Cul⁺ = sqrtS⁺ * U'
    Cdl⁺ = Vt' * sqrtS⁺

    @plansor Pl⁺[-1 -3; -5] := Cul⁺[-1; 2] * Tu[2 -3; 4] * Cul[4; -5]
    @plansor Pl⁻[-1 -3; -5] := Cdl[-1; 2] * Td[2 -3; 4] * Cdl⁺[4; -5]
    
    # for i in 1:10
    #     Cul, Cld, Cul⁺, Cdl⁺, Pl⁺, Pl⁻ = reorthgonal(Cul, Cld, Cul⁺, Cdl⁺, Pl⁺, Pl⁻)
    # end

    return Cul, Cdl, Pl⁺/sqrt(λ), Pl⁻/sqrt(λ)
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
