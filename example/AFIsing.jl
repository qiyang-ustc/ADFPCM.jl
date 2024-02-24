using ADFPCM
using HDF5
using Random
using OMEinsum,LinearAlgebra
using ADFPCM:Emap

let 
    include("exampletensors.jl")
    include("exampleobs.jl")
    d = 2
    χ = 16
    β = 100
    # atype = CuArray
    atype = Array
    
    # rm("./log/eigenconvergence.log")
    # rm("./log/fidelity.log")
    # rm("./log/error.log")

    folder = "./data/AFIsing/"
    model = Ising_Triangle_bad2(1, 1, β)
    M = atype(reshape(model_tensor(model, Val(:Sbulk)), 2,2,2,2))
    
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