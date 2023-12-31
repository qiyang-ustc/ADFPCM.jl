struct FPCMRuntime{MT <: AbstractArray{<:Number, 4}, CT <: AbstractArray{<:Number, 2}, ET <: AbstractArray{<:Number, 3}}
    M::MT
    Cul::CT
    Cld::CT
    Cdr::CT
    Cru::CT
    Tu::ET
    Tl::ET
    Td::ET
    Tr::ET
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
    T = rand!(similar(M,χ,D,χ))
    Params.verbose && println("random initial fpcm_χ$(χ) environment-> ")

    return FPCMRuntime(M, C, C, C, C, T, T, T, T)
end

function FPCMRuntime(M, chkp_file::String, Params)
    rt = loadtype(chkp_file, FPCMRuntime)
    Params.verbose && println("fpcm environment load from $(chkp_file), set up χ=$(Params.χ) is blocked ->") 
    if typeof(M) <: CuArray
        rt = FPCMRuntime(M, CuArray(rt.Cul), CuArray(rt.Cld), CuArray(rt.Cdr), CuArray(rt.Cru), CuArray(rt.Tu), CuArray(rt.Tl), CuArray(rt.Td), CuArray(rt.Tr))
    end  
    return rt
end

cycle(rt::FPCMRuntime) = FPCMRuntime(permutedims(rt.M,(2,3,4,1)), rt.Cld, rt.Cdr, rt.Cru, rt.Cul, rt.Tl, rt.Td, rt.Tr, rt.Tu)

rotatemove = cycle ∘ leftmove
rightmove = leftmove ∘ cycle ∘ cycle
cyclemove = rotatemove ∘ rotatemove ∘ rotatemove ∘ rotatemove
hvmove = cycle ∘ rightmove ∘ leftmove

function initialize_runtime(M, Params)
    in_chkp_file = Params.infolder*"/χ$(Params.χ).h5"
    if isfile(in_chkp_file)                               
        rt = FPCMRuntime(M, in_chkp_file, Params)   
    else
        rt = FPCMRuntime(M, Val(:random), Params)
    end
    return rt
end

FPCM(rt::FPCMRuntime, Params) = FPCM(rt.M, rt, Params)
function FPCM(M, rt::FPCMRuntime, Params)
    freenergy = logZ(rt)
    for i = 1:Params.maxiter
        # rt = cyclemove(rt)
        rt = hvmove(rt)
        freenergy_new = logZ(rt)
        err = abs(freenergy_new - freenergy)
        freenergy = freenergy_new

        nn = Zygote.@ignore nonnormality(rt)

        Zygote.@ignore begin
            Params.verbose && i % Params.output_interval == 0 && print(logentry(i, err, freenergy, nn))

            if Params.ifsave && err < Params.savetol && (i % Params.save_interval == 0 || err < Params.tol)
                rts = FPCMRuntime(Array(M), Array(rt.Cul), Array(rt.Cld), Array(rt.Cdr), Array(rt.Cru), Array(rt.Tu), Array(rt.Tl), Array(rt.Td), Array(rt.Tr))
                isdir(Params.outfolder) || mkdir(Params.outfolder)
                out_chkp_file = Params.outfolder*"/χ$(Params.χ).h5"
                savetype(out_chkp_file, rts, FPCMRuntime)
    
                logfile = open(Params.outfolder*"/χ$(Params.χ).log", "a")
                write(logfile, logentry(i, err, freenergy, nn))
                close(logfile)
            end
        end

        if err < Params.tol && i > Params.miniter
            Params.verbose && Zygote.@ignore print(logentry(i, err, freenergy, nn))
            break
        end
        
    end

    return rt
end

