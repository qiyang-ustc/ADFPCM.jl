using GPUArraysCore: AbstractGPUArray

Base.getindex(xs::AbstractGPUArray{T,0}, I::Integer...) where T = getindex(Array(xs), I...)