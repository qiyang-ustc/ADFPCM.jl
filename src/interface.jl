abstract type Algorithm end

@with_kw mutable struct FPCM <: Algorithm
    χ::VectorSpace
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

@with_kw mutable struct CTMRG <: Algorithm
    χ::VectorSpace
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

function env(M, Params::Algorithm)
    rt = initialize_runtime(M, Params)
    env(rt, Params)
end