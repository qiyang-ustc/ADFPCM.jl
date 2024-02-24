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

function Cenv(Tu, Td, Cl;tol=1E-12)
    λ, cl, info = eigsolve(x -> Cmap(x, Tu, Td), Cl, 1, :LM,tol=tol)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], cl[1]
end

function Eenv(Tu, Td, M, Tl;tol=1E-12)
    λ, al, info = eigsolve(x -> Emap(x, Tu, Td, M), Tl, 1, :LM,tol=tol)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], al[1]
end

function filltwo(Pl⁺, Pl⁻)
    Plm⁺ = reshape(Pl⁺, :, size(Pl⁺,3))
    Plm⁻ = reshape(permutedims(Pl⁻, (3,2,1)), :, size(Pl⁻,1))

    q1,r1 = qr(Plm⁺)
    q2,r2 = qr(Plm⁻)
    q1, q2 = Array(q1),Array(q2)

    @warn "Finding 0 in diagonals, fill with 1.0"
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

    Cul, Cdl, Cul⁺, Cdl⁺, Pl⁺, Pl⁻ = reorthgonal(Cul, Cdl, Cul⁺, Cdl⁺, Pl⁺, Pl⁻)

    PI = ein"pji,ijq->pq"(Pl⁺, Pl⁻)
    PIstability = norm(abs.(PI) - Diagonal(ones(size(PI,1))),1)
    if PIstability > 1E-5
        print(abs.(diag(PI)))
        @warn "PI is not identity:$(PIstability)"
    end
    return Cul, Cdl, Pl⁺, Pl⁻
end

function leftmove(t::Tuple{Float64,FPCMRuntime}) 
    err, rt = t
    derr,rt = leftmove(rt)
    return err+derr, rt
end

function leftmove(rt::FPCMRuntime)
    @unpack M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr = rt
    Cul, Cld, Pl⁺, Pl⁻ = getPL(Tu, Td, Cul*Cld)

    λCul, Cul = Cenv(Tu, Pl⁻, Cul)
    λCld, Cld = Cenv(Pl⁺, Td, Cld)
    λTl, nTl = Eenv(Pl⁺, Pl⁻, M, Tl+rand!(similar(Tl), size(Tl))*1E-7)
    
    err = convergence(nTl,Tl,rt)
    FileIO.open("./log/fidelity.log","a") do fid
        write(fid,"$(err)\n")
    end
    Tl = nTl

    convergence_Cul = norm(Cmap(Cul, Tu, Pl⁻) - λCul*Cul)
    convergence_Cld = norm(Cmap(Cld, Pl⁺, Td) - λCld*Cld)
    convergence_Tl = norm(Emap(Tl, Pl⁺, Pl⁻, M) - λTl*Tl)

    FileIO.open("./log/eigenconvergence.log","a") do fid
        write(fid,"$(convergence_Cul),$(convergence_Cld),$(convergence_Tl)\n")
    end

    return err, FPCMRuntime(M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr)
end