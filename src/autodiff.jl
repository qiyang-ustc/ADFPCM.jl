@non_differentiable save(file, name, object)
@non_differentiable load(file, name)

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

function num_grad(f, a::AbstractTensorMap; δ::Real=1e-5)
    b = Array(copy(convert(Array, a)))
    df = map(CartesianIndices(b)) do i
        foo = x -> (ac = copy(b); ac[i] = x; f(TensorMap(ac, space(a))))
        num_grad(foo, b[i], δ=δ)
    end
    return TensorMap(df, space(a))
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

           ┌──   Tu  ──┐         1 ────┬──── 3 
           │     │     │         │     2     │ 
dTd  = -  Tl  ── M ──  ξl        ├─ 4 ─┼─ 5 ─┤ 
           │     │     │         │     6     │  
           └──       ──┘         7 ────┴──── 8 
```
"""
Ǝmap(x, Tu, Td, M) = @plansor y[-7; -1 -4] := Td[-7 6; 8] * x[8; 3 5] * M[-4 2; 6 5] * Tu[3 2; -1] 

function ChainRulesCore.rrule(::typeof(Eenv), Tu, Td, M, Tl)
    λl, Tl = Eenv(Tu, Td, M, Tl)
    # @show λl
    function back((dλ, dTl))
        dTl -= dot(Tl, dTl) * Tl
        ξl, info = linsolve(R -> Ǝmap(R, Tu, Td, M), dTl', -λl, 1; maxiter = 1)
        @assert info.converged==1
        errL = dot(Tl', ξl)
        abs(errL) > 1e-1 && throw("Tl and ξl aren't orthometric. err = $(errL)")
        @plansor dTu[-1; -3 -2] := -Tl[-1 4; 7] * Td[7 6; 8] * M[4 -2; 6 5] * ξl[8; -3 5]
        @plansor dTd[-8; -7 -6] := -ξl[-8; 3 5] * Tu[3 2; 1] * M[4 2; -6 5] * Tl[1 4; -7]
        @plansor dM[-6 -5; -4 -2] := -Td[7 -6; 8] * ξl[8; 3 -5] * Tl[1 -4; 7] * Tu[3 -2; 1]
        return NoTangent(), dTu', dTd', dM', NoTangent()...
    end
    return (λl, Tl), back
end

"""
    function ChainRulesCore.rrule(::typeof(norm_FL), Tu, Td, C; kwargs...) where {T}

```
           ┌──       ──┐   
           │     │     │   
dTu  = -   C     │     ξl  
           │     │     │   
           └──   Td  ──┘   

           ┌──   Tu  ──┐          1──────┬──────2  
           │     │     │          │      │      │  
dTd  = -   C     │     ξl         │      3      │  
           │     │     │          │      │      │  
           └──       ──┘          4──────┴──────5  
```
"""

Ɔmap(x, Tu, Td) = @plansor y[-4; -1] := Td[-4 3; 5] * x[5; 2] * Tu[2 3; -1]

function ChainRulesCore.rrule(::typeof(Cenv), Tu, Td, C; kwargs...)
    λl, C = Cenv(Tu, Td, C)
    function back((dλ, dC))
        dC -= dot(C, dC) * C
        ξl, info = linsolve(x -> Ɔmap(x, Tu, Td), dC', -λl, 1; maxiter = 1)
        # @show space(ξl) space(Ɔmap(ξl, Tu, Td)) space(dC')
        @assert info.converged==1
        errL = dot(C', ξl)
        abs(errL) > 1e-1 && throw("C and ξl aren't orthometric. err = $(errL)")
        @plansor dTu[-1; -2 -3] := -C[-1; 4] * Td[4 -3; 5] * ξl[5; -2]
        @plansor dTd[-5; -4 -3] := -ξl[-5; 2] * Tu[2 -3; 1] * C[1; -4]
        return NoTangent(), dTu', dTd', NoTangent()...
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