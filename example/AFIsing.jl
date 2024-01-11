using ADFPCM
using ADFPCM: log, overlap
using Random

let
    include("exampletensors.jl")
    include("exampleobs.jl")

    β = 100
    atype = Array
    # alg = CTMRG
    Random.seed!(42)
    for χ in 32:1:32
        model = Ising_Triangle_bad2(1, 1, β)
        M = atype(reshape(model_tensor(model, Val(:Sbulk)), 2,2,2,2))
        # _, _, mcM = mcform(M)
        rt = env(M, FPCM(χ=χ, ifsave=true, verbose=true, maxiter=10, miniter=1, tol =1e-14, infolder="./data/$model/"))
        # rt = env(M, CTMRG(χ=χ,  ifsave=true, verbose=false, maxiter=1000, miniter=100, tol=1e-10, infolder="./data/$model/"))
        # @show 
        @show logZ(rt) - 0.3230659669
        # println("$χ, $(logZ(rt)), ")
        # Au = rt.Tu
        # Ad = rt.Td
        # @show overlap(Au, Ad)
    end
end