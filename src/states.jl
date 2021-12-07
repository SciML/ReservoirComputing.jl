abstract type AbstractStates end
abstract type NonLinearAlgorithm end

#states types
"""
    StandardStates()

No modification of the states takes place, default option.
"""
struct StandardStates <: AbstractStates end

"""
    ExtendedStates()

The states are extended with the input data, for the training section, and the prediction data, 
during the prediction section. This is obtained with a vertical concatenation of the data and the states.
"""
struct ExtendedStates <: AbstractStates end

struct PaddedStates{T} <: AbstractStates
    padding::T
end

struct PaddedExtendedStates{T} <: AbstractStates 
    padding::T
end

"""
    PaddedStates(padding)
    PaddedStates(;padding=1.0)

The states are padded with a chosen value. Usually this value is set to one. The padding is obtained through a 
vertical concatenation of the padding value and the states.
"""
function PaddedStates(;padding=1.0)
    PaddedStates(padding)
end

"""
    PaddedExtendedStates(padding)
    PaddedExtendedStates(;padding=1.0)

The states are extended with the training data or predicted data and subsequently padded with a chosen value. 
Usually the padding value is set to one. The padding and the extension are obtained through a vertical concatenation 
of the padding value, the data and the states.
"""
function PaddedExtendedStates(;padding=1.0)
    PaddedExtendedStates(padding)
end

#functions of the states to apply modifications
function (states_type::StandardStates)(nla_type, x, y)
    nla(nla_type, x)
end

function (states_type::ExtendedStates)(nla_type, x, y)
    x_tmp = vcat(y, x)
    nla(nla_type, x_tmp)
end

function (states_type::PaddedStates)(nla_type, x, y)
    x_tmp = vcat(fill(states_type.padding, (1, size(x, 2))), x)
    nla(nla_type, x_tmp)
end

function (states_type::PaddedExtendedStates)(nla_type, x, y)
    x_tmp = vcat(y, x)
    x_tmp = vcat(fill(states_type.padding, (1, size(x, 2))), x_tmp)
    nla(nla_type, x_tmp)
end

#non linear algorithms
"""
    NLADefault()

Returns the array untouched, default option.
"""
struct NLADefault <: NonLinearAlgorithm end

function nla(::NLADefault, x)
    return x
end

"""
    NLAT1()
Applies the \$ \\text{T}_1 \$ transformation algorithm, as defined in [1] and [2].

[1] Chattopadhyay, Ashesh, et al. "_Data-driven prediction of a multi-scale Lorenz 96 chaotic system using a hierarchy of deep learning methods: 
Reservoir computing, ANN, and RNN-LSTM._" (2019).

[2] Pathak, Jaideep, et al. "_Model-free prediction of large spatiotemporally chaotic systems from data: 
A reservoir computing approach._" Physical review letters 120.2 (2018): 024102.
"""
struct NLAT1 <: NonLinearAlgorithm end

function nla(::NLAT1, x_old)
    x_new = copy(x_old)
    for i =1:size(x_new, 1)
        if mod(i, 2)!=0
            x_new[i,:] = copy(x_old[i,:].*x_old[i,:])
        end
    end
    x_new
end

"""
    NLAT2()
Apply the \$ \\text{T}_2 \$ transformation algorithm, as defined in [1].

[1] Chattopadhyay, Ashesh, et al. "_Data-driven prediction of a multi-scale Lorenz 96 chaotic system using a hierarchy of deep learning methods: 
Reservoir computing, ANN, and RNN-LSTM._" (2019).
"""
struct NLAT2 <: NonLinearAlgorithm end

function nla(::NLAT2, x_old)
    x_new = copy(x_old)
    for i =2:size(x_new, 1)-1
        if mod(i, 2)!=0
            x_new[i,:] = copy(x_old[i-1,:].*x_old[i-2,:])
        end
    end
    x_new
end

"""
    NLAT3()
Apply the \$ \\text{T}_3 \$ transformation algorithm, as defined in [1].

[1] Chattopadhyay, Ashesh, et al. "_Data-driven prediction of a multi-scale Lorenz 96 chaotic system using a hierarchy of deep learning methods: 
Reservoir computing, ANN, and RNN-LSTM._" (2019).
"""
struct NLAT3 <: NonLinearAlgorithm end

function nla(::NLAT3, x_old)
    x_new = copy(x_old)
    for i =2:size(x_new, 1)-1
        if mod(i, 2)!=0
            x_new[i,:] = copy(x_old[i-1,:].*x_old[i+1,:])
        end
    end
    x_new
end
