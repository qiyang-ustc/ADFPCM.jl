using HDF5

function savetype(file, object, type)
    @assert (typeof(object) <: type)
    h5open(file,"w") do fid
        for field in fieldnames(type)
            fid[String(field)] = getfield(object,field)
        end
    end
end

function loadtype(file, type)
    object=Vector{Array}()
    h5open(file,"r") do fid
        for field in fieldnames(type)
            push!(object, Array(fid[String(field)]))
        end
    end
    return type(tuple(object)...)
end