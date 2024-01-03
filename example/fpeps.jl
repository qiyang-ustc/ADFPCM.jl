using ADFPCM
using HDF5
using Random
using OMEinsum,LinearAlgebra
using ADFPCM:Emap

let 
    d = 4
    χ = 2

    Random.seed!(56)
    
    folder = "./data/fpeps/"
    rm("./log/eigenconvergence.log")
    rm("./log/fidelity.log")
    rm("./log/error.log")

    M = load(folder*"/M.h5")["M"]

    # M = CuArray(M)
    _, _, mcM = mcform(M)
    params = ADFPCM.Params(χ=χ, ifsave=true, infolder=folder, maxiter=2000)

    rt = initialize_runtime(mcM, params)
    rt = FPCM(rt, params)

    χstart = χ
    for χ in χstart:1:36
        print("χ=",χ,"\n")
        params = ADFPCM.Params(χ=χ, ifsave=true, infolder=folder, maxiter=3000)
        rt = FPCM(expand(rt,χ,1E-7), params)
        @show nonnormality(rt)
    end

end