export nonnormality, mcform, expand

function logZ(rt::FPCMRuntime)
    @unpack M, Cul, Cld, Au, Al, Ad = rt
    λT , _ = Eenv(Au, Ad, M, ein"ij,jkl,lp->ikp"(Cul,Al,Cul))
    λL , _ = Cenv(Au, Ad, Cul*Cld)
    return log(abs(λT/λL))
end

function nonnormality(rt)
    @unpack M, Au, Ad, Al = rt
    λw, _, _ = eigsolve(x -> Emap(x, Au, Ad, M), Al, 1, :LM)
    λs, _, _ = eigsolve(x -> Emap(Emap(x, Au, Ad, M),permutedims(Au,(3,2,1)),
    permutedims(Ad,(3,2,1)), permutedims(M,(1,4,3,2))), Al, 1, :LM)
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

    @unpack M, Cul, Cld, Cdr, Cru, Au, Al, Ad, Ar = rt
    Cul, Cld, Cdr, Cru = map(C->expand_c(C,χ), (Cul, Cld, Cdr, Cru))
    Au, Al, Ad, Ar = map(A->expand_a(A,χ), (Au, Al, Ad, Ar))

    return FPCMRuntime(M, Cul, Cld, Cdr, Cru, Au, Al, Ad, Ar)
end

logentry(i, err, freenergy, nn) = @sprintf("i = %d,\terr = %.3e,\tlogZ = %.15f,\tnonnormality=%.3f\n", i, err, freenergy, nn)