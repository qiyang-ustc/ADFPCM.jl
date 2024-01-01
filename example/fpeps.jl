using ADFPCM
using HDF5
using Random
using OMEinsum,LinearAlgebra
using ADFPCM:Emap

let 
    d = 4
    χ = 36

    Random.seed!(54)
    
    folder = "./data/fpeps/"
    M = load(folder*"/M.h5")["M"]
    # M = CuArray(M)
    _, _, mcM = mcform(M)
    params = ADFPCM.Params(χ=χ, ifsave=false, infolder=folder, maxiter=3000)

    rt = initialize_runtime(mcM, params)
    rt = FPCM(rt, params)

    # for i in 18:2:32
    #     print("χ=",i,"\n")
    #     params = ADFPCM.Params(χ=i, ifsave=true, infolder=folder, maxiter=10000)
    #     rt = FPCM(expand(rt,i,1E-7), params)
    #     @show nonnormality(rt)
    # end

end