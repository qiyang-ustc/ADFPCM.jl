using ADFPCM
using ADFPCM: logZ

let
    include("exampletensors.jl")
    include("exampleobs.jl")

    β = 100
    atype = Array
    # alg = CTMRG 
    alg = FPCM
    for χ in 2 .^ (4:4)
        model = Ising_Triangle_bad2(1, 1, β)
        M = atype(reshape(model_tensor(model, Val(:Sbulk)), 2,2,2,2))
        _, _, mcM = mcform(M)
        params = alg(χ=χ, ifsave=true, maxiter=1000, tol =1e-10, infolder="./data/$model/$alg/")
        
        rt = initialize_runtime(mcM, params)
        rt = env(rt, params)
        # @show logZ(rt) - 0.3230659669
        println("$χ, $(logZ(rt)), ")
    end
end