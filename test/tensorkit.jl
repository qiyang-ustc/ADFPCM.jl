using TensorKit
using TensorKit: ×
using KrylovKit
using Test
using Random

@testset "initial Tensor" begin
    A = Tensor(randn, ℝ^3 ⊗ ℝ^2 ⊗ ℝ^4) # if it use randn(3,2,4)?
    
    A′ = Tensor(randn, ℝ^3) ⊗ Tensor(randn, ℝ^2) ⊗ Tensor(randn, ℝ^4) # should they be equal?
    @test space(A) == space(A′) 

    A = Tensor(randn, ComplexF64, ℂ^3 ⊗ ℂ^2 ⊗ ℂ^4)

    A = TensorMap(randn, ComplexF64, ℂ^2 ⊗ ℂ^2, ℂ^2)
    B = TensorMap(randn, ComplexF64, ℂ^2 ⊗ ℂ^2, ℂ^2)

    D = TensorMap(randn, ComplexF64, (ℂ^2)' ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2)
end

@testset "tensor product" begin
    v = Tensor(randn, ℝ^3)
    m1 = TensorMap(randn, ℝ^4, ℝ^3)
    w = m1 * v
    v′ = v ⊗ v
    m1′ = m1 ⊗ m1
    w′ = m1′ * v′
    @test w′ ≈ w ⊗ w

    A = TensorMap(randn, ComplexF64, ℂ^2 ⊗ ℂ^2, ℂ^2)
    B = TensorMap(randn, ComplexF64, ℂ^2 ⊗ ℂ^2, ℂ^2)

    D = TensorMap(randn, ComplexF64, (ℂ^2)' ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2)
    @tensor C[-1 5;2 1] := A[-1 5;4]*B[4 2;1]
    @plansor C′[-1 5 2;1] := A[-1 5;4]*B[4 2;1]
    @plansor E[-1 -2;1 3] := C[-1 5 2;1]*D[5 -2;2 3]
    # @test C ≈ C′
    
    @plansor C = A'[3;1 2] * B[1 2; 3]
    @test dot(A,B) ≈ C

    @test convert(Array, A) ≈ reshape(A.data,2,2,2)
end

@testset "svd" begin
    A = Tensor(randn, ℝ^3 ⊗ ℝ^2 ⊗ ℝ^4)
    U, S, Vd = tsvd(A, (1,3), (2,))
    @tensor A′[a,b,c] := U[a,c,d]*S[d,e]*Vd[e,b]

    @test A ≈ A′
    @test U'*U ≈ one(U'*U) # left isometric tensor
    @test Vd*Vd' ≈ one(Vd*Vd') # right isometric tensor

    P = U*U' 
    @test P*P ≈ P # should be a projector

    @tensor A2′′[a b; c] := U[a,c,d]*S[d,e]*Vd[e,b];
    @test space(A2′′) == ((ℝ^3 ⊗ ℝ^2) ← ℝ^4)
end

@testset "Z2 symmetry" begin
    V1 = ℤ₂Space(0=>3,1=>2)
    V2 = ℤ₂Space(0=>1,1=>1)
    A = TensorMap(randuniform, V1*V1,V2')
    B = TensorMap(randuniform, V1'*V1,V2)

    @tensor C[a,b] := A[a,c,d]*B[c,b,d]
    @plansor C′[a,b] := A[a,c,d]*B[c,b,d]
    @test C ≈ C′

    tA = convert(Array, A)
    ttA = TensorMap(tA, V1*V1,V2')
    @test ttA ≈ A
end

@testset "U1 symmetry" begin
    V = U₁Space(0=>2,1=>1,-1=>1)
    A = TensorMap(randn, V*V, V)
    B = TensorMap(randn, V*V, V)

    @test Rep[U₁](0=>3,1=>2,-1=>1) == U1Space(-1=>1,1=>2,0=>3)
end

@testset "mix" begin 
    V = Rep[U₁×ℤ₂]((0, 0) => 2, (1, 1) => 1, (-1, 0) => 1)
    A = TensorMap(randn, V*V, V)
end

@testset "fuse" begin
    V1 = ℤ₂Space(0=>3,1=>2)
    V2 = ℤ₂Space(0=>1,1=>1)
    A = TensorMap(randn, ComplexF64, V1*V2*V1*V2, V1*V2)
    B = TensorMap(A.data, fuse(V1*V2)*fuse(V1*V2), V1*V2)
    I = isomorphism(fuse(V1*V2)*fuse(V1*V2), V1*V2*V1*V2)
    @tensor C[-1 -2; -3 -4] := I[-1 -2; 1 2 3 4] * A[1 2 3 4; -3 -4]
    @test norm(A) ≈ norm(B) ≈ norm(C)
    @test B ≈ C

    D = ℤ₂Space(0=>3,1=>2)
    d = ℤ₂Space(0=>1,1=>1)
    ipeps = TensorMap(randn, ComplexF64, D*D*d, D*D)
    @tensor M[-2 -1 -4 -3; -6 -5 -8 -7] := ipeps'[-6 -8; -2 -4 9] * ipeps[-1 -3 9;-5 -7]
    It = isomorphism(fuse(D', D), D'*D)

    @tensor Mt[-1 -2; -3 -4] := It[-1; 2 1] * It[-2; 4 3] * It'[6 5; -3] * It'[8 7; -4] * M[2 1 4 3; 6 5 8 7]
    Mt2 = TensorMap(M.data, fuse(D'*D)*fuse(D'*D), fuse(D'*D)'*fuse(D'*D)')
    @test norm(M) ≈ norm(Mt) ≈ norm(Mt2)
    
    @test Mt.data != Mt2.data
end