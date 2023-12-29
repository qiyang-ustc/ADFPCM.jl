using ADFPCM
using Test

include("exampletensors.jl")
include("exampleobs.jl")

@testset "J1-J2 with with $atype " for Ni = [1], Nj = [1], atype = [CuArray]
    d = 2
    χ = 128

    β = 1.299
    model = J1J2_1(Ni, Nj, 1.0, 0.5, β)
    M = atype(reshape(model_tensor(model, Val(:bulk)), 2,2,2,2))
    rt = FPCM(M, ADFPCM.Params(χ=χ, infolder = "./data/$model"))
    @show logZ(rt)
    f1 = logZ(rt)
    f2 = (logZ ∘ cycle)(rt)
    f3 = (logZ ∘ cycle ∘ cycle)(rt)
    f4 = (logZ ∘ cycle ∘ cycle ∘ cycle)(rt)
    @show f1 f2 f3 f4
end