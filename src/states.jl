abstract type AbstractStates end
abstract type AbstractPaddedStates <: AbstractStates end
abstract type NonLinearAlgorithm end

function pad_state!(states_type::AbstractPaddedStates, x_pad, x)
    x_pad = vcat(fill(states_type.padding, (1, size(x, 2))), x)
    return x_pad
end

function pad_state!(states_type, x_pad, x)
    x_pad = x
    return x_pad
end

#states types
"""
    StandardStates()

No modification of the states takes place, default option.
"""
struct StandardStates <: AbstractStates end

"""
    ExtendedStates()

The states are extended with the input data, for the training section, and the prediction
data, during the prediction section. This is obtained with a vertical concatenation of the
data and the states.
"""
struct ExtendedStates <: AbstractStates end

struct PaddedStates{T} <: AbstractPaddedStates
    padding::T
end

struct PaddedExtendedStates{T} <: AbstractPaddedStates 
    padding::T
end

"""
    PaddedStates(padding)
    PaddedStates(;padding=1.0)

The states are padded with a chosen value. Usually this value is set to one. The padding is obtained through a 
vertical concatenation of the padding value and the states.
"""
function PaddedStates(;padding=1.0)
    return PaddedStates(padding)
end

"""
    PaddedExtendedStates(padding)
    PaddedExtendedStates(;padding=1.0)

The states are extended with the training data or predicted data and subsequently padded with a chosen value. 
Usually the padding value is set to one. The padding and the extension are obtained through a vertical concatenation 
of the padding value, the data and the states.
"""
function PaddedExtendedStates(;padding=1.0)
    return PaddedExtendedStates(padding)
end

#functions of the states to apply modifications
function (::StandardStates)(nla_type, x, y)
    return nla(nla_type, x)
end

function (::ExtendedStates)(nla_type, x, y)
    x_tmp = vcat(y, x)
    return nla(nla_type, x_tmp)
end

function (states_type::PaddedStates)(nla_type, x, y)
    x_tmp = vcat(fill(states_type.padding, (1, size(x, 2))), x)
    return nla(nla_type, x_tmp)
end

function (states_type::PaddedExtendedStates)(nla_type, x, y)
    x_tmp = vcat(y, x)
    x_tmp = vcat(fill(states_type.padding, (1, size(x, 2))), x_tmp)
    return nla(nla_type, x_tmp)
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

function nla(nla_type, x_old)
    x_new = similar(x_old)
    nla!(nla_type, x_old, x_new)
    return x_new
end

"""
    NLAT1()
Applies the \$ \\text{T}_1 \$ transformation algorithm, as defined in [1] and [2].

[1] Chattopadhyay, Ashesh, et al. "_Data-driven prediction of a multi-scale Lorenz 96
chaotic system using a hierarchy of deep learning methods: Reservoir computing,
ANN, and RNN-LSTM._" (2019).

[2] Pathak, Jaideep, et al. "_Model-free prediction of large spatiotemporally chaotic
systems from data: A reservoir computing approach._"
Physical review letters 120.2 (2018): 024102.
"""
struct NLAT1 <: NonLinearAlgorithm end

function nla!(::NLAT1, x_old, x_new)
    x_new[2:2:end, :] = x_old[2:2:end, :]
    x_new[1:2:end, :] = x_old[1:2:end, :].^2
end

"""
    NLAT2()
Apply the \$ \\text{T}_2 \$ transformation algorithm, as defined in [1].

[1] Chattopadhyay, Ashesh, et al. "_Data-driven prediction of a multi-scale Lorenz 96
chaotic system using a hierarchy of deep learning methods: Reservoir computing, ANN,
and RNN-LSTM._" (2019).
"""
struct NLAT2 <: NonLinearAlgorithm end

function nla!(::NLAT2, x_old, x_new)
    x_new[1, :] = x_old[1, :]
    x_new[2:2:end, :] = x_old[2:2:end, :]
    x_new[3:2:end-1, :] = x_old[2:2:end-2, :].*x_old[1:2:end-3, :]
end

"""
    NLAT3()
Apply the \$ \\text{T}_3 \$ transformation algorithm, as defined in [1].

[1] Chattopadhyay, Ashesh, et al. "_Data-driven prediction of a multi-scale Lorenz 96
chaotic system using a hierarchy of deep learning methods: Reservoir computing, ANN,
and RNN-LSTM._" (2019).
"""
struct NLAT3 <: NonLinearAlgorithm end

function nla!(::NLAT3, x_old, x_new)
    x_new[1,:]= x_old[1, :]
    x_new[2:2:end, :]= x_old[2:2:end, :]
    x_new[3:2:end-1, :]= x_old[2:2:end-2, :].*x_old[4:2:end, :]
end
