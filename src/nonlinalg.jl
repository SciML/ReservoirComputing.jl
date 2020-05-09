function NonLinAlgDefault(x_old::AbstractArray{T}) where T<: AbstractFloat
    x_new = copy(x_old)
    return x_new
end

function NonLinAlgT1(x_old::AbstractArray{T}) where T<: AbstractFloat
    x_new = copy(x_old)
    for i =1:size(x_new, 1)
        if mod(i, 2)!=0
            x_new[i, :] = copy(x_old[i,:].*x_old[i,:])
        end
    end
    return x_new
end

function NonLinAlgT2(x_old::AbstractArray{T}) where T<: AbstractFloat
    x_new = copy(x_old)
    for i =2:size(x_new, 1)-1
        if mod(i, 2)!=0
            x_new[i, :] = copy(x_old[i-1,:].*x_old[i-2,:])
        end
    end
    return x_new
end

function NonLinAlgT3(x_old::Array{T}) where T<: AbstractFloat
    x_new = copy(x_old)
    for i =2:size(x_new, 1)-1
        if mod(i, 2)!=0
            x_new[i, :] = copy(x_old[i-1,:].*x_old[i+1,:])
        end
    end
    return x_new
end
