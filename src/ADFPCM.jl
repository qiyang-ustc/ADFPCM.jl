module ADFPCM
    using Reexport
    using ChainRulesCore
    using Zygote

    export FPCMRuntime, FPCM, initialize_runtime

    @reexport using CUDA, LinearAlgebra, OMEinsum, KrylovKit
    @reexport using Parameters, Random, HDF5, FileIO, Printf

    include("FPCMRuntime.jl")
    include("environment.jl")
    include("fpcm.jl")
    include("utils.jl")
    include("h5api.jl")
    include("cudapatch.jl")
    include("autodiff.jl")
    include("interface.jl")
end
