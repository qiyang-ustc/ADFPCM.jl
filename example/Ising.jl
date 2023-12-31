using ADFPCM
using ADFPCM: cycle, logZ, logZ2
using Zygote

let
    include("exampletensors.jl")
    include("exampleobs.jl")
    d = 2
    χ = 16
    atype = Array

    function LogZ(β) 
        model = Ising(1, 1, β)
        M = atype(reshape(model_tensor(model, Val(:bulk)), 2,2,2,2))
        logZ2(FPCM(M, ADFPCM.Params(χ=χ, ifsave=true, maxiter=1000, miniter=10, 
                                   infolder="./data/$model/")))
    end
  
    @show LogZ(0.5) - 1.0257928172049902

    @show Zygote.gradient(LogZ, 0.5)[1] - 1.745564581767667
    
end