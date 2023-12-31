export nonnormality, mcform, expand

function logZ(rt::FPCMRuntime)
    @unpack M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr = rt

    # alternative implementation easier but slow 
    # λM, _ = Eenv(Tu, Td, M, ein"ij,jkl,lp->ikp"(Cul,Tl,Cul))
    # λN, _ = Cenv(Tu, Td, Cul*Cld)

    E = ein"(ab,bcd),de->ace"(Cul,Tl,Cld)
    ∃ = ein"(ab,bcd),de->ace"(Cdr,Tr,Cru)
    田 = ein"abc,cba->"(Emap(E, Tu, Td, M), ∃)[]
    日 = ein"abc,cba->"(E,∃)[]
    λM = 田/日

    C = ein"ab,bc->ac"(Cul,Cld)
    Ɔ = ein"ab,bc->ac"(Cdr,Cru)
    日 = ein"ab,ba->"(Cmap(C, Tu, Td), Ɔ)[]
    口 = ein"ab,ba->"(C,Ɔ)[]
    λN = 日/口

    return log(abs(λM/λN))
end

function nonnormality(rt)
    # Very costful calculation
    @unpack M, Cul, Cld, Tu, Td, Tl = rt
    
    Cul, Cld, Pl⁺, Pl⁻ = getPL(Tu, Td, Cul*Cld)
    λw, _, _ = eigsolve(x -> Emap(x, Pl⁺, Pl⁻, M), Tl, 1, :LM)
    λs, _, _ = eigsolve(x -> Emap(Emap(x, Pl⁺, Pl⁻, M),
    permutedims(Pl⁺,(3,2,1)),
    permutedims(Pl⁻,(3,2,1)), permutedims(M,(1,4,3,2))), Tl, 1, :LM)
    return sqrt(abs(λs[1]))/abs(λw[1])
end


function mcform(M)
    aM = Array(M)
    x = ein"ijil->jl"(aM)
    _, vh = eigen(x)
    aM = ein"aj,(ijkl,lb)->iakb"(inv(vh),aM,vh)
    y = ein"ijkj->ik"(aM)
    _, vv = eigen(y)
    aM = ein"(ai,ijkl),kb->ajbl"(inv(vv),aM,vv)
    aM = typeof(M)(aM)
    return vv, vh, aM
end    

function expand(rt,χ,ϵ=1E-7)
    function expand_c(C,χ)
        C⁺ = ϵ*randn(eltype(C),χ,χ)
        C⁺[1:size(C,1),1:size(C,2)] .= C
        return typeof(C)(C⁺)
    end

    function expand_a(A,χ)
        A⁺ = ϵ*randn(eltype(A),χ,size(A,2),χ)
        A⁺[1:size(A,1),1:size(A,2),1:size(A,3)] .= A
        return typeof(A)(A⁺)
    end

    @unpack M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr = rt
    Cul, Cld, Cdr, Cru = map(C->expand_c(C,χ), (Cul, Cld, Cdr, Cru))
    Tu, Tl, Td, Tr = map(A->expand_a(A,χ), (Tu, Tl, Td, Tr))

    return FPCMRuntime(M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr)
end

logentry(i, err, freenergy, nn) = @sprintf("i = %5d,\terr = %.3e,\tlogZ = %.15f,\tnonnormality=%.3f\n", i, err, freenergy, nn)