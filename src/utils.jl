export nonnormality, mcform, expand

function logZ(rt::Runtime)
    @unpack M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr = rt

    # alternative implementation easier but slow 
    # @plansor E[-1 -2; -3] := Cul[-1; 1] * Tl[1 -2; 2] * Cld[2; -3]  
    # λM, _ = Eenv(Tu, Td, M, E)
    # λN, _ = Cenv(Tu, Td, Cul*Cld)

    @plansor E[-1 -3; -5] := Cul[-1; 2] * Tl[2 -3; 4] * Cld[4; -5]
    @plansor Ǝ[-1 -3; -5] := Cdr[-1; 2] * Tr[2 -3; 4] * Cru[4; -5]
    @plansor 田 = Emap(E, Tu, Td, M)[1 2; 3] * Ǝ[3 2; 1]
    @plansor 日 = E[1 2; 3] * Ǝ[3 2; 1]
    λM = 田/日

    @plansor C[-1; -3] := Cul[-1; 2] * Cld[2; -3]
    @plansor Ɔ[-1; -3] := Cdr[-1; 2] * Cru[2; -3]
    @plansor 日 = Cmap(C, Tu, Td)[1; 2] * Ɔ[2; 1]
    @plansor 口 = C[1; 2] * Ɔ[2; 1]
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

    return Runtime(M, Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr)
end

function overlap(Au, Ad)
    χ = size(Au,1)
    C = rand(χ,χ)
    nu = eigsolve(L -> ein"(ad,bca),ecd -> be"(L,Au,conj(Au)), C, 1, :LM)[1][1]
    Au /= sqrt(nu)

    nd = eigsolve(L -> ein"(ad,acb),dce -> be"(L,Ad,conj(Ad)), C, 1, :LM)[1][1]
    Ad /= sqrt(nd)

    nud = eigsolve(L -> ein"(ad,bca),dce -> be"(L,Au,conj(Ad)), C, 1, :LM)[1][1]
    abs(nud)
end


logentry(i, err, freenergy) = @sprintf("i = %5d,\terr = %.3e,\tlogZ = %.15f\n", i, err, freenergy)