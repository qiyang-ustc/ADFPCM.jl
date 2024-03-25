using ADFPCM
using ADFPCM: cycle, logZ

let
    include("exampletensors.jl")
    include("exampleobs.jl")
    d = 4
    χ = 16
    β = 100
    atype = Array
    
    model = Ising_Triangle_bad(1, 1, β)
    M = atype(reshape(model_tensor(model, Val(:Sbulk)), d,d,d,d))

    _, _, mcM = mcform(M)
    params = ADFPCM.Params(χ=χ, ifsave=true, maxiter=1000, infolder="./data/$model/")
    
    rt = initialize_runtime(mcM, params)
    rt = FPCM(rt, params)
    @show logZ(rt) - 0.3230659669
    f1 = logZ(rt)
    f2 = (logZ ∘ cycle)(rt)
    f3 = (logZ ∘ cycle ∘ cycle)(rt)
    f4 = (logZ ∘ cycle ∘ cycle ∘ cycle)(rt)
    @show f1 f2 f3 f4
end