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
    atype = typeof(M)
    if atype <: Array
        C = rand!(similar(M,χ,χ))
        T = rand!(similar(M,χ,D,χ))
    else
        qnD, qnχ, dimsD, dimsχ = alg.U1info
        indqn = [qnχ, qnD, qnD, qnχ]
        indims = [dimsχ, dimsD, dimsD, dimsχ]

        C = Zygote.@ignore randinitial(M, χ, χ; 
        dir = [1, -1], indqn = [qnχ, qnχ], indims = [dimsχ, dimsχ])

        # T = randinitial(M, χ, D, χ; 
        # dir = [1, 1, -1], indqn = [qnχ, qnD, qnχ], indims = [dimsχ, dimsD, dimsχ])
        T = Zygote.@ignore symmetryreshape(randinitial(M, χ, Int(sqrt(D)), Int(sqrt(D)), χ; 
        dir = [1, -1, 1, -1], indqn = indqn, indims = indims
        ), 
        χ, D, χ; reinfo = (nothing, nothing, nothing, indqn, indims, nothing, nothing))[1] # for double-layer ipeps
    end

    alg.verbose && printstyled("start $alg random initial fpcm_χ$(χ) environment->  \n"; bold=true, color=:green) 
    

    return Runtime(M, C, C, C, C, T, T, T, T)
end

function Runtime(M, chkp_file::String, alg)
    rt = load(chkp_file, "env")
    Zygote.@ignore alg.verbose && printstyled("start $alg environment load from $(chkp_file), set up χ=$(alg.χ) is blocked -> \n"; bold=true, color=:green) 
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
cyclemove(rt, alg) = foldl(rotatemove, repeat([alg], 4), init=rt)
hvmove(rt, alg) = cycle(rightmove(leftmove(rt, alg), alg))
randmove(rt, alg) = rand([cycle, cycle ∘ cycle ∘ cycle])(leftmove(rt, alg))

function initialize_runtime(M, alg)
    in_chkp_file = alg.infolder*"/χ$(alg.χ).jld2"
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

            if alg.ifsave && err < alg.savetol && i % alg.save_interval == 0
                rts = Runtime(Array(M), Array(rt.Cul), Array(rt.Cld), Array(rt.Cdr), Array(rt.Cru), Array(rt.Tu), Array(rt.Tl), Array(rt.Td), Array(rt.Tr))
                ispath(alg.outfolder) || mkpath(alg.outfolder)
                out_chkp_file = alg.outfolder*"/χ$(alg.χ).jld2"
                save(out_chkp_file, "env", rts)
    
                logfile = open(alg.outfolder*"/χ$(alg.χ).log", "a")
                write(logfile, logentry(i, err, freenergy))
                close(logfile)
            end
        end

        if err < alg.tol && i > alg.miniter
            alg.verbose && Zygote.@ignore print(logentry(i, err, freenergy))
            Zygote.@ignore begin 
                if alg.ifsave && err < alg.savetol
                rts = Runtime(Array(M), Array(rt.Cul), Array(rt.Cld), Array(rt.Cdr), Array(rt.Cru), Array(rt.Tu), Array(rt.Tl), Array(rt.Td), Array(rt.Tr))
                ispath(alg.outfolder) || mkpath(alg.outfolder)
                out_chkp_file = alg.outfolder*"/χ$(alg.χ).jld2"
                save(out_chkp_file, "env", rts)

                logfile = open(alg.outfolder*"/χ$(alg.χ).log", "a")
                write(logfile, logentry(i, err, freenergy))
                close(logfile)
                end
            end
            break
        end
        
    end

    return rt
end

