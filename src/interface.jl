@with_kw mutable struct Params
    χ::Int = 16
    tol::Float64 = 1e-14
    maxiter::Int = 1000
    miniter::Int = 100
    output_interval::Int = 1
    ifsave::Bool = true
    savetol::Float64 = 1e-1
    save_interval::Int = 10
    infolder = "./data/"
    outfolder = infolder
    verbose::Bool = true
end

function FPCM(M, Params)
    rt = initialize_runtime(M, Params)
    FPCM(rt, Params)
end