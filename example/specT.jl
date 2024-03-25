using ADFPCM
using ADFPCM: cycle, logZ

χ = 20
atype = Array

M = zeros(ComplexF64,2,2,2,2)
M[2,1,1,1]=1.0
M[1,2,1,1]=1.0
M[2,2,1,1]=1.0
M[1,2,2,2]=1.0
M[2,1,2,2]=1.0
M[1,1,2,2]=1.0

_, _, mcM = mcform(M)

filename = "χ$(χ).h5"
rt = ADFPCM.loadtype("./data/AFIsing/"*filename, FPCMRuntime)

@unpack M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr = rt

moveL = reshape(ein"bca,dce->bead"(Tu,Td),χ^2,χ^2)
moveR = reshape(ein"bca,dce->adbe"(Tu,Td),χ^2,χ^2)

# U,S,V = svd(moveL)
wL, vL = eigen(moveL)
wR, vR = eigen(moveR)

# _, _, Pu, Pd = ADFPCM.getPL(Tu,Td,Cul)

# movePL = reshape(ein"bca,dce->adbe"(Pu,Pd),χ^2,χ^2)
# Up,Sp,Vp = svd(movePL)
# wp, vp = eigen(movePL)

# Cmap(x, Tu, Td) = ein"(bca,ad),dce->be"(Tu, x, Td)
# result_p = eigsolve(x -> Cmap(x, Pu, Pd), rand(ComplexF64,χ,χ), 1, :LM;)
# result_t = eigsolve(x -> Cmap(x, Tu, Td), rand(ComplexF64,χ,χ), 1, :LM;)

# v and vp, which one has smaller nonnormality?
# opn_v = opnorm(inv(v))*opnorm(v)/maximum(abs, w)^2
# opn_vp = opnorm(inv(vp))*opnorm(vp)/maximum(abs, wp)^2

# canonical measure:
# λ(t) = eigsolve(x -> Cmap(x, t, permutedims(conj.(t),(3,2,1))), rand(ComplexF64,χ,χ), 1, :LM;)[1][1]
# λTu = abs(λ(Tu))
# λTd = abs(λ(Td))
# λPu = abs(λ(Pu))
# λPd = abs(λ(Pd))

# @show abs(result_p[1][1])
# @show abs(result_t[1][1])
# @show abs(result_p[1][1]/sqrt(λPu*λPd))
# @show abs(result_t[1][1]/sqrt(λTu*λTd))

# left right bi-orthogonal
# iCmap(x, Tu, Td) = ein"(bca,be),dce->ad"(Tu, x, Td)
# movePR = reshape(ein"bca,dce->bead"(Pu,Pd),χ^2,χ^2)

# wR, vR = eigen(moveR)
# wpR, vpR = eigen(movePR)

@show abs(dot(vL[:,1],vR[:,1]))
@show abs(dot(vp[:,1],vpR[:,1]))

function lrpre(l,r)
    l,r = reshape(l,χ,χ),reshape(r,χ,χ)

    u = l*transpose(r)
    wu, vu = eigen(u)

    l = inv(vu)*l
    r = transpose(vu)*r

    d = transpose(l) * r
    wd, vd = eigen(d)

    l = l * transpose(inv(vd))
    r = r * vd

    return l, r, vu, vd
end

l, r, vu, vd = lrpre(vL[:,1],vR[:,1])
@show abs(dot(l,r)/sqrt(dot(l,l)*dot(r,r)))

# Plru = ein"(pl,lkj),ji->pki"(vu,Tu,inv(vu))
# Plrd = ein"(ij,jkl),lp->ikp"(inv(transpose(vd)),Td,transpose(vd))

Plru = ein"(pl,lkj),ji->pki"(inv(vu),Tu,vu)
Plrd = ein"(ij,jkl),lp->ikp"(transpose(vd),Td,inv(transpose(vd)))

movePlrL = reshape(ein"bca,dce->bead"(Plru,Plrd),χ^2,χ^2)
UlrL,SlrL,VlrL = svd(movePlrL)
wlrL, vlrL = eigen(movePlrL)

movePlrR = reshape(ein"bca,dce->adbe"(Plru,Plrd),χ^2,χ^2)
UlrR,SlrR,VlrR = svd(movePlrR)
wlrR, vlrR = eigen(movePlrR)

l,r = vlrL[:,1],vlrR[:,1]
@show abs(dot(l,r)/sqrt(dot(l,l)*dot(r,r)))