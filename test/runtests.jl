using ADFPCM
using Test

@testset "ADFPCM.jl" begin
    @testset "environment.jl" begin
        println("environment tests running...")
        include("environment.jl")
    end
    @testset "fpcmruntime.jl" begin
        println("fpcmruntime tests running...")
        include("fpcmruntime.jl")
    end
end
