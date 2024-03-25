using ADFPCM
using ADFPCM: cycle, logZ
using Random

let
    χ = 1
    atype = Array
    Random.seed!(1234)
    if isfile("./data/AFIsing/χ$(χ).h5")
        rm("./data/AFIsing/χ$(χ).h5") # refresh cache
    end

    M = zeros(ComplexF64,2,2,2,2)
    M[2,1,1,1]=1.0
    M[1,2,1,1]=1.0
    M[2,2,1,1]=1.0
    M[1,2,2,2]=1.0
    M[2,1,2,2]=1.0
    M[1,1,2,2]=1.0
    
    # _, _, mcM = mcform(M)
    mcM = M
    params = ADFPCM.Params(χ=χ, ifsave=true, maxiter=1000,
    save_interval=1, infolder="./data/AFIsing/")
    
    rt = initialize_runtime(mcM, params)
    rt = FPCM(rt, params)

    @show logZ(rt) - 0.3230659669
    f1 = logZ(rt)
    f2 = (logZ ∘ cycle)(rt)
    f3 = (logZ ∘ cycle ∘ cycle)(rt)
    f4 = (logZ ∘ cycle ∘ cycle ∘ cycle)(rt)
    @show f1 f2 f3 f4
end