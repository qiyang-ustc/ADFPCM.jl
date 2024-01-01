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
    params = ADFPCM.Params(χ=χ, ifsave=true, infolder=folder, maxiter=1000)

    rt = initialize_runtime(mcM, params)
    rt = FPCM(rt, params)
end