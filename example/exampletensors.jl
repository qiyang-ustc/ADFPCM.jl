using OMEinsum
using Zygote
using LinearAlgebra: I

const isingβc = log(1+sqrt(2))/2

abstract type HamiltonianModel end

export Ising, IsingU, Ising_Triangle_bad, Ising_Triangle_bad2, Ising_Triangle_good, J1J2_1, J1J2_2
export model_tensor

struct Ising <: HamiltonianModel 
    Ni::Int
    Nj::Int
    β::Float64
end

struct IsingU <: HamiltonianModel 
    Ni::Int
    Nj::Int
    β::Float64
    τ::Float64
end

struct Ising_Triangle_bad <: HamiltonianModel 
    Ni::Int
    Nj::Int
    β::Float64
end

struct Ising_Triangle_bad2 <: HamiltonianModel 
    Ni::Int
    Nj::Int
    β::Float64
end
struct Ising_Triangle_good <: HamiltonianModel 
    Ni::Int
    Nj::Int
    β::Float64
end

struct Ising_Triangle_good2 <: HamiltonianModel 
    Ni::Int
    Nj::Int
    β::Float64
end

struct J1J2_1 <: HamiltonianModel 
    Ni::Int
    Nj::Int
    J1::Float64
    J2::Float64
    β::Float64
end

struct J1J2_2 <: HamiltonianModel 
    Ni::Int
    Nj::Int
    J1::Float64
    J2::Float64
    β::Float64
end

J1J2_1(Ni, Nj, J2, β) = J1J2_1(Ni, Nj, 1.0, J2, β)
J1J2_2(Ni, Nj, J2, β) = J1J2_2(Ni, Nj, 1.0, J2, β)

"""
    model_tensor(model::Ising, type)

return the  `MT <: HamiltonianModel` `type` tensor at inverse temperature `β` for  two-dimensional
square lattice tensor-network.
"""
function model_tensor(model::Ising, ::Val{:bulk})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = Zygote.@ignore ComplexF64[-1. 1;1 -1]
    w = exp.(- β * ham)
    wsq = sqrt(w)
    m = ein"ia,ib,ic,id -> abcd"(wsq, wsq, wsq, wsq)

    M = Zygote.Buffer(m, 2,2,2,2,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::IsingU, ::Val{:bulk})
    Ni, Nj, β, τ = model.Ni, model.Nj, model.β, model.τ
    ham = Zygote.@ignore ComplexF64[-1. 1;1 -1]
    w = exp.(- β * ham)
    wsq = sqrt(w)
    m = ein"ia,ib,ic,id -> abcd"(wsq, wsq, wsq, wsq)

    U = exp(-τ * [1 0; 0 -1])
    m = ein"abcd,bi,dj -> aicj"(m,U,U^-1)

    M = Zygote.Buffer(m, 2,2,2,2,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::Ising_Triangle_bad, ::Val{:bulk})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = Zygote.@ignore ComplexF64[1. -1;-1 1]
    w = exp.(- β * ham)

    m = reshape(ein"mi,mj,mk,ml,mn,mo,qp->ijklponq"(I(2),I(2),I(2),I(2),w,w,w),4,4,4,4)
    M = Zygote.Buffer(m, 4,4,4,4,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::Ising_Triangle_bad2, ::Val{:bulk})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = Zygote.@ignore -ones(ComplexF64, 2,2,2)
    ham[1,1,1] = ham[2,2,2] = 3
    w = exp.(- β * ham)

    m = reshape(ein"ijk,kl,nl,ml->ijnm"(w,I(2),I(2),I(2)),2,2,2,2)
    M = Zygote.Buffer(m, 2,2,2,2,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::J1J2_1, ::Val{:bulk})
    Ni, Nj, J1, J2, β = model.Ni, model.Nj, model.J1, model.J2, model.β
    m = Zygote.@ignore zeros(ComplexF64, 2,2,2,2)
    for i in -1:2:1
        for j in -1:2:1
            for k in -1:2:1
                for l in -1:2:1
                    x,y,z,w = (i+3)÷2, (j+3)÷2, (k+3)÷2, (l+3)÷2
                    m[x,y,z,w] += exp(β/2*(i*j+i*l+k*j+k*l)-β*J2/J1*(i*k+j*l))
                end
            end
        end
    end

    M = Zygote.Buffer(m, 2,2,2,2,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::J1J2_2, ::Val{:bulk})
    Ni, Nj, J1, J2, β = model.Ni, model.Nj, model.J1, model.J2, model.β
    m = Zygote.@ignore zeros(ComplexF64, 4,4,4,4)
    for i in -1:2:1
        for j in -1:2:1
            for k in -1:2:1
                for l in -1:2:1
                    a,b,c,d = (i+1)÷2,(j+1)÷2,(k+1)÷2,(l+1)÷2
                    x,y,z,w = a*2+b+1,b*2+c+1,d*2+c+1,a*2+d+1
                    m[x,y,z,w] = exp(β/2*(i*j+i*l+k*j+k*l)-β*J2/J1*(i*k+j*l))
                end
            end
        end
    end

    M = Zygote.Buffer(m, 4,4,4,4,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::J1J2_2, ::Val{:energy})
    Ni, Nj, J1, J2, β = model.Ni, model.Nj, model.J1, model.J2, model.β
    m = Zygote.@ignore zeros(ComplexF64, 4,4,4,4)
    for i in -1:2:1
        for j in -1:2:1
            for k in -1:2:1
                for l in -1:2:1
                    a,b,c,d = (i+1)÷2,(j+1)÷2,(k+1)÷2,(l+1)÷2
                    x,y,z,w = a*2+b+1,b*2+c+1,d*2+c+1,a*2+d+1
                    m[x,y,z,w] = (-(i*j+i*l+k*j+k*l)/2 + J2/J1*(i*k+j*l)) * exp(β/2*(i*j+i*l+k*j+k*l)-β*J2/J1*(i*k+j*l))
                end
            end
        end
    end

    M = Zygote.Buffer(m, 4,4,4,4,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end


"""
    residual entropy
"""
function model_tensor(model::Ising_Triangle_bad, ::Val{:Sbulk})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = Zygote.@ignore ComplexF64[4/3. -2/3;-2/3 4/3]
    w = exp.(- β * ham)

    m = reshape(ein"mi,mj,mk,ml,mn,mo,qp->ijklponq"(I(2),I(2),I(2),I(2),w,w,w),4,4,4,4)
    M = Zygote.Buffer(m, 4,4,4,4,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::Ising_Triangle_bad2, ::Val{:Sbulk})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = Zygote.@ignore zeros(ComplexF64, 2,2,2)
    ham[1,1,1] = ham[2,2,2] = 4
    w = exp.(- β * ham)

    m = reshape(ein"ijk,kl,nl,ml->ijnm"(w,I(2),I(2),I(2)),2,2,2,2)
    M = Zygote.Buffer(m, 2,2,2,2,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::Ising_Triangle_good, ::Val{:bulk})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = Zygote.@ignore ComplexF64[1. -1;-1 1]
    w1 = exp.(- β * ham)
    w2 = exp.(- β * ham/2)

    m = reshape(ein"im,in,jo,jp,kq,kr,ls,lt,ik,kl,lj,ji,il->mqrsptno"(I(2),I(2),I(2),I(2),I(2),I(2),I(2),I(2),w2,w2,w2,w2,w1),4,4,4,4)
    M = Zygote.Buffer(m, 4,4,4,4,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

"""
    residual entropy
"""
function model_tensor(model::Ising_Triangle_good, ::Val{:Sbulk})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = Zygote.@ignore ComplexF64[4/3. -2/3;-2/3 4/3]
    w1 = exp.(- β * ham)
    w2 = exp.(- β * ham/2)

    m = reshape(ein"im,in,jo,jp,kq,kr,ls,lt,ik,kl,lj,ji,il->mqrsptno"(I(2),I(2),I(2),I(2),I(2),I(2),I(2),I(2),w2,w2,w2,w2,w1),4,4,4,4)
    M = Zygote.Buffer(m, 4,4,4,4,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

"""
    residual entropy
"""
function model_tensor(model::Ising_Triangle_good2, ::Val{:Sbulk})
    Ni, Nj, β = model.Ni, model.Nj, model.β

    m = Zygote.@ignore zeros(ComplexF64, 2,2,2)
    for i in -1:2:1
        for j in -1:2:1
            for k in -1:2:1
                a,b,c = (i+3)÷2,(j+3)÷2,(k+3)÷2
                m[a,b,c] = (1 + i*j*k) / 2 * exp(-β/2*(i+j+k+1))
            end
        end
    end

    m = ein"abc,cde->abde"(m,m)
    M = Zygote.Buffer(m, 2,2,2,2,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::Ising, ::Val{:mag})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    a = reshape(ComplexF64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 -1], 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    m = ein"abcd,ai,bj,ck,dl -> ijkl"(a,q,q,q,q)
    M = Zygote.Buffer(m, 2,2,2,2,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::Ising, ::Val{:energy})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = ComplexF64[-1 1;1 -1]
    w = exp.(-β .* ham)
    we = ham .* w
    wsq = sqrt(w)
    wsqi = wsq^(-1)
    e = (ein"ai,im,bm,cm,dm -> abcd"(wsqi,we,wsq,wsq,wsq) + ein"am,bi,im,cm,dm -> abcd"(wsq,wsqi,we,wsq,wsq) + 
        ein"am,bm,ci,im,dm -> abcd"(wsq,wsq,wsqi,we,wsq) + ein"am,bm,cm,di,im -> abcd"(wsq,wsq,wsq,wsqi,we)) / 2
    M = Zygote.Buffer(e, 2,2,2,2,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = e
    end
    return copy(M)
end

function model_tensor(model::Ising_Triangle_bad, ::Val{:energy})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = Zygote.@ignore ComplexF64[1. -1;-1 1]
    w = exp.(- β * ham)
    we = ham .* w

    m = reshape(ein"mi,mj,mk,ml,mn,mo,qp->ijklponq"(I(2),I(2),I(2),I(2),we, w , w ),4,4,4,4) + 
        reshape(ein"mi,mj,mk,ml,mn,mo,qp->ijklponq"(I(2),I(2),I(2),I(2),w , we, w ),4,4,4,4) +
        reshape(ein"mi,mj,mk,ml,mn,mo,qp->ijklponq"(I(2),I(2),I(2),I(2),w , w , we),4,4,4,4)
    M = Zygote.Buffer(m, 4,4,4,4,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::Ising_Triangle_good, ::Val{:energy})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = Zygote.@ignore ComplexF64[1. -1;-1 1]
    w1 = exp.(- β * ham)
    w2 = exp.(- β * ham/2)
    we1 = ham .* w1
    we2 = ham .* w2/2

    m = reshape(ein"im,in,jo,jp,kq,kr,ls,lt,ik,kl,lj,ji,il->mqrsptno"(I(2),I(2),I(2),I(2),I(2),I(2),I(2),I(2),
        we2,w2,w2,w2,w1),4,4,4,4) + 
        reshape(ein"im,in,jo,jp,kq,kr,ls,lt,ik,kl,lj,ji,il->mqrsptno"(I(2),I(2),I(2),I(2),I(2),I(2),I(2),I(2),w2,we2,w2,w2,w1),4,4,4,4) +
        reshape(ein"im,in,jo,jp,kq,kr,ls,lt,ik,kl,lj,ji,il->mqrsptno"(I(2),I(2),I(2),I(2),I(2),I(2),I(2),I(2),w2,w2,we2,w2,w1),4,4,4,4) +
        reshape(ein"im,in,jo,jp,kq,kr,ls,lt,ik,kl,lj,ji,il->mqrsptno"(I(2),I(2),I(2),I(2),I(2),I(2),I(2),I(2),w2,w2,w2,we2,w1),4,4,4,4) +
        reshape(ein"im,in,jo,jp,kq,kr,ls,lt,ik,kl,lj,ji,il->mqrsptno"(I(2),I(2),I(2),I(2),I(2),I(2),I(2),I(2),w2,w2,w2,w2,we1),4,4,4,4) 
    M = Zygote.Buffer(m, 4,4,4,4,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

