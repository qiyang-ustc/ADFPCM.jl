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

function obs_env(M, Params::Algorithm)
    rt = initialize_runtime(M, Params)
    Env(env(rt, Params))
end
struct Env{ET <: AbstractTensorMap{<:IndexSpace, 2,1}}
    Eul::ET
    Eur::ET
    Edl::ET
    Edr::ET
    Elu::ET
    Eld::ET
    Elo::ET
    Eru::ET
    Erd::ET
    Ero::ET
    function Env(rt::Runtime)
        @unpack Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr = rt
        @plansor Elu[-1 -3; -4] := Cul[-1; 2] * Tl[2 -3; -4]
        @plansor Eld[-1 -2; -4] := Tl[-1 -2; 3] * Cld[3; -4]
        @plansor Elo[-1 -3; -5] := Cul[-1; 2] * Tl[2 -3; 4] * Cld[4; -5]
        @plansor Erd[-1 -3; -4] := Cdr[-1; 2] * Tr[2 -3; -4]
        @plansor Eru[-1 -2; -4] := Tr[-1 -2; 3] * Cru[3; -4]
        @plansor Ero[-1 -3; -5] := Cdr[-1; 2] * Tr[2 -3; 4] * Cru[4; -5]
        ET = typeof(Tu)
        new{ET}(Tu, Tu, Td, Td, Elu, Eld, Elo, Eru, Erd, Ero)
    end
end