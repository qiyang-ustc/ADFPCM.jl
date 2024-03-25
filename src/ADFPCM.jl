module ADFPCM
    using Reexport
    using ChainRulesCore
    using Zygote
    using CUDA

    export FPCMRuntime, FPCM, initialize_runtime

    @reexport using LinearAlgebra, OMEinsum, KrylovKit
    @reexport using Parameters, Random, HDF5, FileIO, Printf

    include("environment.jl")
    include("fpcmruntime.jl")
    include("utils.jl")
    include("h5api.jl")
    include("cudapatch.jl")
    include("autodiff.jl")
    include("interface.jl")
end
