struct Runtime{MT <: AbstractArray{<:Number, 4}, CT <: AbstractArray{<:Number, 2}, ET <: AbstractArray{<:Number, 3}}
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

function Runtime(M, ::Val{:random}, alg)
    χ = alg.χ
    D = size(M,1)
    C = rand!(similar(M,χ,χ))
    T = rand!(similar(M,χ,D,χ))
    alg.verbose && Zygote.@ignore printstyled("start $alg random initial fpcm_χ$(χ) environment->  \n"; bold=true, color=:green) 
    

    return Runtime(M, C, C, C, C, T, T, T, T)
end

function Runtime(M, chkp_file::String, alg)
    rt = loadtype(chkp_file, Runtime)
    alg.verbose && Zygote.@ignore printstyled("start $alg environment load from $(chkp_file), set up χ=$(alg.χ) is blocked -> \n"; bold=true, color=:green) 
    if typeof(M) <: CuArray
        rt = Runtime(M, CuArray(rt.Cul), CuArray(rt.Cld), CuArray(rt.Cdr), CuArray(rt.Cru), CuArray(rt.Tu), CuArray(rt.Tl), CuArray(rt.Td), CuArray(rt.Tr))
    else
        rt = Runtime(M, rt.Cul, rt.Cld, rt.Cdr, rt.Cru, rt.Tu, rt.Tl, rt.Td, rt.Tr)
    end  
    return rt
end

cycle(rt::Runtime) = Runtime(permutedims(rt.M,(2,3,4,1)), rt.Cld, rt.Cdr, rt.Cru, rt.Cul, rt.Tl, rt.Td, rt.Tr, rt.Tu)
rotatemove(rt, alg) = cycle(leftmove(rt, alg))
rightmove(rt, alg) = leftmove(cycle(cycle(rt)), alg)
# cyclemove(rt, alg) = foldl(rotatemove, repeat([alg], 4), init=rt) # have bugs for AD https://github.com/JuliaDiff/ChainRules.jl/pull/569
cyclemove(rt, alg) = rotatemove(rotatemove(rotatemove(rotatemove(rt, alg), alg), alg), alg)
hvmove(rt, alg) = cycle(rightmove(leftmove(rt, alg), alg))
randmove(rt, alg) = rand([cycle, cycle ∘ cycle, cycle ∘ cycle ∘ cycle])(leftmove(rt, alg))

function initialize_runtime(M, alg)
    in_chkp_file = alg.infolder*"/χ$(alg.χ).h5"
    if isfile(in_chkp_file)                               
        rt = Runtime(M, in_chkp_file, alg)   
    else
        rt = Runtime(M, Val(:random), alg)
    end
    return rt
end

function env(rt::Runtime, alg::Algorithm)
    M = rt.M
    freenergy = logZ(rt)
    for i = 1:alg.maxiter
        rt = cyclemove(rt, alg)
        # rt = cyclemove(rt, CTMRG(χ=alg.χ))
        # rt = hvmove(rt, alg)
        # rt = randmove(rt, alg)
        freenergy_new = logZ(rt)
        err = abs(freenergy_new - freenergy)
        freenergy = freenergy_new


        Zygote.@ignore begin
            alg.verbose && i % alg.output_interval == 0 && print(logentry(i, err, freenergy))

            if alg.ifsave && err < alg.savetol && (i % alg.save_interval == 0 || err < alg.tol)
                rts = Runtime(Array(M), Array(rt.Cul), Array(rt.Cld), Array(rt.Cdr), Array(rt.Cru), Array(rt.Tu), Array(rt.Tl), Array(rt.Td), Array(rt.Tr))
                ispath(alg.outfolder) || mkpath(alg.outfolder)
                out_chkp_file = alg.outfolder*"/χ$(alg.χ).h5"
                savetype(out_chkp_file, rts, Runtime)
    
                logfile = open(alg.outfolder*"/χ$(alg.χ).log", "a")
                write(logfile, logentry(i, err, freenergy))
                close(logfile)
            end
        end

        if err < alg.tol && i > alg.miniter
            alg.verbose && Zygote.@ignore print(logentry(i, err, freenergy))
            break
        end
        
    end

    return rt
end

