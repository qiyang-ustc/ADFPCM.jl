import Base.Array

Array(A::AbstractTensorMap) = (A.data.values .= Array(A.data.values); A)
CuArray(A::AbstractTensorMap) = (A.data.values .= CuArray(A.data.values); A)

regularform(S::String) = replace(S, "=>" => "→")
rejoinpath(S::String...) = regularform(Base.joinpath(S))
@non_differentiable rejoinpath(S...)