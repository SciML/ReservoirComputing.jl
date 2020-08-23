struct NLADefault{T<:AbstractFloat} <: NonLinearAlgorithm end

"""
    NLADefault()
Return the array untouched, default option.
"""
NLADefault() = NLADefault{Float64}()

nla(::NLADefault{T}, x::AbstractArray{T}) where T<: AbstractFloat = x

struct NLAT1{T<:AbstractFloat} <: NonLinearAlgorithm end
"""
    NLAT1()
Apply the \$ \\text{T}_1 \$ transformation algorithm, as defined in [1] and [2]

[1] Chattopadhyay, Ashesh, et al. "Data-driven prediction of a multi-scale Lorenz 96 chaotic system using a hierarchy of deep learning methods: Reservoir computing, ANN, and RNN-LSTM." (2019).

[2] Pathak, Jaideep, et al. "Model-free prediction of large spatiotemporally chaotic systems from data: A reservoir computing approach." Physical review letters 120.2 (2018): 024102.
"""
NLAT1() = NLAT1{Float64}()
nla(::NLAT1{T}, x::AbstractArray{T}) where T<: AbstractFloat = _nlat1(x)

struct NLAT2{T<:AbstractFloat} <: NonLinearAlgorithm end
"""
    NLAT2()
Apply the \$ \\text{T}_2 \$ transformation algorithm, as defined in [1]

[1] Chattopadhyay, Ashesh, et al. "Data-driven prediction of a multi-scale Lorenz 96 chaotic system using a hierarchy of deep learning methods: Reservoir computing, ANN, and RNN-LSTM." (2019).
"""
NLAT2() = NLAT2{Float64}()
nla(::NLAT2{T}, x::AbstractArray{T}) where T<: AbstractFloat = _nlat2(x)

struct NLAT3{T<:AbstractFloat} <: NonLinearAlgorithm end
"""
    NLAT3()
Apply the \$ \\text{T}_3 \$ transformation algorithm, as defined in [1]

[1] Chattopadhyay, Ashesh, et al. "Data-driven prediction of a multi-scale Lorenz 96 chaotic system using a hierarchy of deep learning methods: Reservoir computing, ANN, and RNN-LSTM." (2019).
"""
NLAT3() = NLAT3{Float64}()
nla(::NLAT3{T}, x::AbstractArray{T}) where T<: AbstractFloat = _nlat3(x)


function _nlat1(x_old::AbstractArray{T}) where T<: AbstractFloat
    x_new = copy(x_old)
    for i =1:size(x_new, 1)
        if mod(i, 2)!=0
            x_new[i, :] = copy(x_old[i,:].*x_old[i,:])
        end
    end
    return x_new
end

function _nlat2(x_old::AbstractArray{T}) where T<: AbstractFloat
    x_new = copy(x_old)
    for i =2:size(x_new, 1)-1
        if mod(i, 2)!=0
            x_new[i, :] = copy(x_old[i-1,:].*x_old[i-2,:])
        end
    end
    return x_new
end

function _nlat3(x_old::Array{T}) where T<: AbstractFloat
    x_new = copy(x_old)
    for i =2:size(x_new, 1)-1
        if mod(i, 2)!=0
            x_new[i, :] = copy(x_old[i-1,:].*x_old[i+1,:])
        end
    end
    return x_new
end
