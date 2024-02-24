using ADFPCM
using HDF5
using Random
using OMEinsum,LinearAlgebra
using ADFPCM:Emap

let 
    
    Random.seed!(1234)
    M = ones(2,2,2,2)
    M = M + 0.1* rand(2,2,2,2)
    
    d = 2
    χ = 16
    # atype = CuArray
    atype = Array
    M = atype(M)

    folder = "./data/fpeps/"
    rm("./log/eigenconvergence.log")
    rm("./log/fidelity.log")
    rm("./log/error.log")
    
    _, _, mcM = mcform(M)
    params = ADFPCM.Params(χ=χ, ifsave=true, infolder=folder, maxiter=1000)
    rt = initialize_runtime(mcM, params)
    rt = FPCM(rt, params)

    χstart = χ
    for χ in χstart+1:1:128
        print("χ=",χ,"\n")
        params = ADFPCM.Params(χ=χ, ifsave=true, infolder=folder, maxiter=1000)
        rt = FPCM(expand(rt,χ,1E-4), params)
        @show nonnormality(rt)
    end
    # 0.3230659669
end