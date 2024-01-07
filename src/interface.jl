abstract type Algorithm end

@with_kw mutable struct FPCM <: Algorithm
    χ::Int = 16
    tol::Float64 = 1e-10
    maxiter::Int = 100
    miniter::Int = 10
    output_interval::Int = 10
    ifsave::Bool = true
    savetol::Float64 = 1e-1
    save_interval::Int = 10
    infolder = "./data/"
    outfolder = infolder
    verbose::Bool = true
    U1info = nothing
end

@with_kw mutable struct CTMRG <: Algorithm
    χ::Int = 16
    tol::Float64 = 1e-10
    maxiter::Int = 100
    miniter::Int = 10
    output_interval::Int = 10
    ifsave::Bool = true
    savetol::Float64 = 1e-1
    save_interval::Int = 10
    infolder = "./data/"
    outfolder = infolder
    verbose::Bool = true
    U1info = nothing
end

function env(M, Params::Algorithm)
    rt = initialize_runtime(M, Params)
    env(rt, Params)
end