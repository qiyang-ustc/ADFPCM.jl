using ADFPCM
using ADFPCM: cyclemove, cycle, logZ
using Test

include("exampletensors.jl")
include("exampleobs.jl")

@testset "AFIsing with with $atype " for Ni = [1], Nj = [1], atype = [Array]
    d = 2
    χ = 16

    β = 10
    model = Ising_Triangle_bad2(Ni, Nj, β)
    M = reshape(atype(model_tensor(model, Val(:Sbulk))),2,2,2,2)
    # initialization
    C = rand(χ,χ)
    A = rand(χ,d,χ)

    Cul, Cld, Cdr, Cru, Au, Al, Ad, Ar = C,C,C,C,A,A,A,A
    state = Cul, Cld, Cdr, Cru, Au, Al, Ad, Ar, M
    # if CUDA.functional()
    #     state = map(CuArray,state)
    # end

    for i = 1:1000
        state = cyclemove(state)
        @show logZ(state), logZ(cycle(state)),  logZ(cycle(cycle(state))), logZ((cycle ∘cycle ∘ cycle)(state))
    end
end
