Array(A::AbstractTensorMap) = (A.data .= Array(A.data); A)
CuArray(A::AbstractTensorMap) = (A.data .= CuArray(A.data); A)