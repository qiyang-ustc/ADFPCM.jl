@non_differentiable savetype(file, object, type)
@non_differentiable loadtype(file, type)

num_grad(f, K::Complex; δ::Real=1e-5) = (f(K + δ / 2) - f(K - δ / 2)) / δ + (f(K + δ / 2 * 1.0im) - f(K - δ / 2 * 1.0im)) / δ * 1.0im
num_grad(f, K::Real; δ::Real=1e-5) = (f(K + δ / 2) - f(K - δ / 2)) / δ

function num_grad(f, a::AbstractArray; δ::Real=1e-5)
    b = Array(copy(a))
    atype = typeof(a)
    df = map(CartesianIndices(b)) do i
        foo = x -> (ac = copy(b); ac[i] = x; f(atype(ac)))
        num_grad(foo, b[i], δ=δ)
    end
    return atype(df)
end

# patch since it's currently broken otherwise
function ChainRulesCore.rrule(::typeof(Base.typed_hvcat), ::Type{T}, rows::Tuple{Vararg{Int}}, xs::S...) where {T,S}
    y = Base.typed_hvcat(T, rows, xs...)
    function back(ȳ)
        return NoTangent(), NoTangent(), NoTangent(), permutedims(ȳ)...
    end
    return y, back
end

# improves performance compared to default implementation, also avoids errors
# with some complex arrays
function ChainRulesCore.rrule(::typeof(LinearAlgebra.norm), A::AbstractArray)
    n = norm(A)
    function back(Δ)
        return NoTangent(), Δ .* A ./ (n + eps(0f0)), NoTangent()
    end
    return n, back
end

function ChainRulesCore.rrule(::typeof(Base.sqrt), A::AbstractArray)
    As = Base.sqrt(A)
    function back(dAs)
        dA =  As' \ dAs ./2 
        return NoTangent(), dA
    end
    return As, back
end

"""
    ChainRulesCore.rrule(::typeof(Eenv), Tu, Td, M, Tl)

```
           ┌──   Tu  ──┐ 
           │     │     │ 
dM    = - Tl ──    ──  ξl
           │     │     │ 
           └──   Td  ──┘ 

           ┌──       ──┐   
           │     │     │   
dTu  = -  Tl ──  M ──  ξl  
           │     │     │   
           └──   Td  ──┘   

           ┌──   Tu  ──┐          a ────┬──── c 
           │     │     │          │     b     │ 
dTd  = -  Tl  ── M ──  ξl         ├─ d ─┼─ e ─┤ 
           │     │     │          │     g     │  
           └──       ──┘          f ────┴──── h 
```
"""

function ChainRulesCore.rrule(::typeof(Eenv), Tu, Td, M, Tl)
    λl, Tl = Eenv(Tu, Td, M, Tl)
    # @show λl
    function back((dλ, dTl))
        dTl -= ein"abc,abc ->"(conj(Tl), dTl)[] * Tl
        ξl, info = linsolve(R -> ein"((cba,ceh),bdge),fgh -> adf"(Tu, R, M, Td), conj(dTl), -λl, 1; maxiter = 1)
        @assert info.converged==1
        # errL = ein"abc,abc ->"(conj(Tl), ξl)[]
        # abs(errL) > 1e-1 && throw("Tl and ξl aren't orthometric. err = $(errL)")
        dTu = -ein"((adf,fgh),bdge),ceh -> cba"(Tl, Td, M, ξl) 
        dTd = -ein"((adf,cba),bdge),ceh -> fgh"(Tl, Tu, M, ξl)
        dM = -ein"(adf,cba),(fgh,ceh) -> bdge"(Tl, Tu, Td, ξl)
        return NoTangent(), conj(dTu), conj(dTd), conj(dM), NoTangent()...
    end
    return (λl, Tl), back
end

"""
    function ChainRulesCore.rrule(::typeof(norm_FL), Tu, Td, C; kwargs...) where {T}

```
           ┌──       ──┐   
           │     │     │   
dTu  = -   C ────┼──── ξl  
           │     │     │   
           └──   Td  ──┘   

           ┌──   Tu  ──┐           a──────┬──────b 
           │     │     │           │      │      │ 
dTd  = -   C  ───┼───  ξl          │      c      │ 
           │     │     │           │      │      │ 
           └──       ──┘           d──────┴──────e 
```
"""

function ChainRulesCore.rrule(::typeof(Cenv), Tu, Td, C; kwargs...)
    λl, C = Cenv(Tu, Td, C)
    function back((dλ, dC))
        dC -= Array(ein"ab,ab ->"(conj(C), dC))[] * C
        ξl, info = linsolve(R -> ein"(be,bca),dce -> ad"(R,Tu,Td), conj(dC), -λl, 1; maxiter = 1)
        @assert info.converged==1
        # errL = ein"ab,ab ->"(conj(C), ξl)[]
        # abs(errL) > 1e-1 && throw("C and ξl aren't orthometric. err = $(errL)")
        dTu = -ein"(ad,dce),be -> bca"(C, Td, ξl) 
        dTd = -ein"(ad,bca),be -> dce"(C, Tu, ξl)
        return NoTangent(), conj(dTu), conj(dTd), NoTangent()...
    end
    return (λl, C), back
end


Zygote.@adjoint function LinearAlgebra.svd(A)
    res = LinearAlgebra.svd(A)
    res, function (dy)
        dU, dS, dVt = dy
        return (svd_back(res.U, res.S, res.V, dU, dS, dVt === nothing ? nothing : dVt'),)
    end
end

struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a
Base.:-(a, zero::ZeroAdder) = a
Base.:-(zero::ZeroAdder, a) = -a
Base.:-(zero::ZeroAdder) = zero

"""
    svd_back(U, S, V, dU, dS, dV)

adjoint for SVD decomposition.

References:
    https://j-towns.github.io/papers/svd-derivative.pdf
    https://giggleliu.github.io/2019/04/02/einsumbp.html
"""
function svd_back(U::AbstractArray, S::AbstractArray{T}, V, dU, dS, dV; η::Real=1e-40) where T
    all(x -> x isa Nothing, (dU, dS, dV)) && return nothing
    η = T(η)
    NS = length(S)
    S2 = S .^ 2
    Sinv = @. S/(S2+η)
    F = S2' .- S2
    F ./= (F.^ 2 .+ η)

    res = ZeroAdder()
    if !(dU isa Nothing)
        UdU = U'*dU
        J = F.*(UdU)
        res += (J+J')*LinearAlgebra.Diagonal(S) + LinearAlgebra.Diagonal(1im*imag(LinearAlgebra.diag(UdU)) .* Sinv)
    end
    if !(dV isa Nothing)
        VdV = V'*dV
        K = F.*(VdV)
        res += LinearAlgebra.Diagonal(S) * (K+K')
    end
    if !(dS isa Nothing)
        res += LinearAlgebra.Diagonal(dS)
    end

    res = U*res*V'

    if !(dU isa Nothing) && size(U, 1) != size(U, 2)
        res += (dU - U* (U'*dU)) * LinearAlgebra.Diagonal(Sinv) * V'
    end

    if !(dV isa Nothing) && size(V, 1) != size(V, 2)
        res = res + U * LinearAlgebra.Diagonal(Sinv) * (dV' - (dV'*V)*V')
    end
    res
end