using ADFPCM
using JLD2
using Random
using OMEinsum,LinearAlgebra
using ADFPCM:Emap

let 
    d = 4
    χ = 16

    Random.seed!(54)
    M = load("./data/M.jld2")["M"]
    # M = CuArray(M)
    _, _, mcM = mcform(M)
    params = ADFPCM.Params(χ=χ, ifsave=false,maxiter=1000)

    rt = initialize_runtime(mcM, params)
    rt = FPCM(rt, params)

    @show nonnormality(rt)
    for i in 18:2:32
        print("χ=",i,"\n")
        params = ADFPCM.Params(χ=χ, ifsave=false,maxiter=10000)
        rt = FPCM(expand(rt,i,1E-7), params)
        @show nonnormality(rt)
    end

end