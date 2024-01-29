import Base.Array

Array(A::AbstractTensorMap) = (A.data .= Array(A.data); A)
CuArray(A::AbstractTensorMap) = (A.data .= CuArray(A.data); A)

regularform(S::String) = replace(S, "=>" => "→")
rejoinpath(S::String...) = regularform(Base.joinpath(S))