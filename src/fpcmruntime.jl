struct FPCMRuntime{MT <: AbstractArray{<:Number, 4}, CT <: AbstractArray{<:Number, 2}, ET <: AbstractArray{<:Number, 3}}
    M::MT
    Cul::CT
    Cld::CT
    Cdr::CT
    Cru::CT
    Au::ET
    Al::ET
    Ad::ET
    Ar::ET
end

@with_kw mutable struct Params
    χ::Int = 16
    tol::Float64 = 1e-14
    maxiter::Int = 1000
    miniter::Int = 100
    output_interval::Int = 1
    ifsave::Bool = true
    savetol::Float64 = 1e-1
    save_interval::Int = 10
    infolder = "./data/"
    outfolder = infolder
    verbose::Bool = true
end

function FPCMRuntime(M, ::Val{:random}, Params)
    χ = Params.χ
    D = size(M,1)
    C = rand!(similar(M,χ,χ))
    A = rand!(similar(M,χ,D,χ))
    Params.verbose && println("random initial fpcm_χ$(χ) environment-> ")

    return FPCMRuntime(M, C, C, C, C, A, A, A, A)
end

function FPCMRuntime(M, chkp_file::String, Params)
    rt = loadtype(chkp_file, FPCMRuntime)
    Params.verbose && println("fpcm environment load from $(chkp_file), set up χ=$(χ) is blocked ->") 
    if typeof(M) <: CuArray
        rt = FPCMRuntime(M, CuArray(rt.Cul), CuArray(rt.Cld), CuArray(rt.Cdr), CuArray(rt.Cru), CuArray(rt.Au), CuArray(rt.Al), CuArray(rt.Ad), CuArray(rt.Ar))
    end  
    return rt
end

cycle(rt::FPCMRuntime) = FPCMRuntime(permutedims(rt.M,(2,3,4,1)), rt.Cld, rt.Cdr, rt.Cru, rt.Cul, rt.Al, rt.Ad, rt.Ar, rt.Au)

rotatemove = cycle ∘ leftmove
rightmove = leftmove ∘ cycle ∘ cycle
cyclemove = rotatemove ∘ rotatemove ∘ rotatemove ∘ rotatemove
hvmove = cycle ∘ rightmove ∘ leftmove


function logZ(rt::FPCMRuntime)
    @unpack M, Cul, Cld, Au, Al, Ad = rt
    λT , _ = Eenv(Au, Ad, M, ein"ij,jkl,lp->ikp"(Cul,Al,Cul))
    λL , _ = Cenv(Au, Ad, Cul*Cld)
    return log(abs(λT/λL))
end

function initialize_runtime(M, Params)
    in_chkp_file = Params.infolder*"/χ$(Params.χ).h5"
    if isfile(in_chkp_file)                               
        rt = FPCMRuntime(M, in_chkp_file, Params)   
    else
        rt = FPCMRuntime(M, Val(:random), Params)
    end
    return rt
end
 

function FPCM(rt, Params)
    @unpack M = rt
    freenergy = logZ(rt)
    for i = 1:Params.maxiter
        # rt = cyclemove(rt)
        rt = hvmove(rt)
        freenergy_new = logZ(rt)
        err = abs(freenergy_new - freenergy)
        freenergy = freenergy_new
        nn = nonnormality(rt)

        i % Params.output_interval == 0 && print(logentry(i, err, freenergy, nn))
        if Params.ifsave && err < Params.savetol && (i % Params.save_interval == 0 || err < Params.tol)
            rts = FPCMRuntime(Array(M), Array(rt.Cul), Array(rt.Cld), Array(rt.Cdr), Array(rt.Cru), Array(rt.Au), Array(rt.Al), Array(rt.Ad), Array(rt.Ar))
            out_chkp_file = Params.outfolder*"/χ$(Params.χ).h5"
            savetype(out_chkp_file, rts, FPCMRuntime)

            logfile = open(Params.outfolder*"/χ$(Params.χ).log", "a")
            write(logfile, logentry(i, err, freenergy, nn))
            close(logfile)
        end

        if err < Params.tol && i > Params.miniter
            print(logentry(i, err, freenergy, nn))
            break
        end
    end

    return rt
end

