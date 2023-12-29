include("fpcm.jl")

d = 2
χ = 64

M = zeros((2,2,2,2))
M[2,1,1,1]=1.0
M[1,2,1,1]=1.0
M[2,2,1,1]=1.0
M[1,2,2,2]=1.0
M[2,1,2,2]=1.0
M[1,1,2,2]=1.0
# logZ = 0.3230659669

# M = zeros((2,2,2,2))
# M[2,1,1,1]=1.0
# M[1,2,1,1]=1.0
# M[1,1,2,1]=1.0
# M[1,1,1,2]=1.0
# logZ = 0.29152163577 # (0.29155024471215657 # passed)

# initialization
C = rand(χ,χ)
A = rand(χ,d,χ)

Cul, Cld, Cdr, Cru, Au, Al, Ad, Ar = C,C,C,C,A,A,A,A
state = Cul, Cld, Cdr, Cru, Au, Al, Ad, Ar, M
if CUDA.functional()
    state = map(CuArray,state)
end

for i = 1:1000
    state = cyclemove(state)
    @show logZ(state), logZ(cycle(state)),  logZ(cycle(cycle(state))), logZ((cycle ∘cycle ∘ cycle)(state))
end