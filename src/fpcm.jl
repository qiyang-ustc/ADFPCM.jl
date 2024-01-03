rotatemove = cycle ∘ leftmove
rightmove = leftmove ∘ cycle ∘ cycle
cyclemove = rotatemove ∘ rotatemove ∘ rotatemove ∘ rotatemove
hvmove = cycle ∘ rightmove ∘ leftmove

FPCM(rt::FPCMRuntime, Params) = FPCM(rt.M, rt, Params)
function FPCM(M, rt::FPCMRuntime, Params)
    freenergy = logZ(rt)
    for i = 1:Params.maxiter
        err, rt = cyclemove(rt)
        # err, rt = hvmove(rt)
        freenergy = logZ(rt)

        FileIO.open("./log/error.log","a") do fid
            write(fid,"$(err)\n")
            write(fid,"$(err)\n")
        end

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