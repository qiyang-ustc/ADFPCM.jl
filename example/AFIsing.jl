using ADFPCM
using ADFPCM: cycle, logZ

let
    include("exampletensors.jl")
    include("exampleobs.jl")
    d = 2
    χ = 16
    β = 100
    atype = Array
    
    model = Ising_Triangle_bad2(1, 1, β)
    M = atype(reshape(model_tensor(model, Val(:Sbulk)), 2,2,2,2))
    _, _, mcM = mcform(M)
    params = ADFPCM.Params(χ=χ, ifsave=false, maxiter=1000)
    
    rt = initialize_runtime(mcM, params)
    rt = FPCM(rt, params)
    @show logZ(rt) - 0.3230659669
    f1 = logZ(rt)
    f2 = (logZ ∘ cycle)(rt)
    f3 = (logZ ∘ cycle ∘ cycle)(rt)
    f4 = (logZ ∘ cycle ∘ cycle ∘ cycle)(rt)
    @show f1 f2 f3 f4
end