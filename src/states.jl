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

When this struct is employed, the states of the reservoir are not modified. It represents the default behavior 
in scenarios where no specific state modification is required. This approach is ideal for applications 
where the inherent dynamics of the reservoir are sufficient, and no external manipulation of the states 
is necessary. It maintains the original state representation, ensuring that the reservoir's natural properties 
are preserved and utilized in computations.
"""
struct StandardStates <: AbstractStates end

"""
    ExtendedStates()

The `ExtendedStates` struct is used to extend the reservoir states by 
vertically concatenating the input data (during training) and the prediction data (during the prediction phase). 
This method enriches the state representation by integrating external data, enhancing the model's capability 
to capture and utilize complex patterns in both training and prediction stages.
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

Creates an instance of the `PaddedStates` struct with specified padding value.
This padding is typically set to 1.0 by default but can be customized.
The states of the reservoir are padded by vertically concatenating this padding value,
enhancing the dimensionality and potentially improving the performance of the reservoir computing model.
This function is particularly useful in scenarios where adding a constant baseline to the states is necessary
for the desired computational task.
"""
function PaddedStates(; padding = 1.0)
    return PaddedStates(padding)
end

"""
    PaddedExtendedStates(padding)
    PaddedExtendedStates(;padding=1.0)

Constructs a `PaddedExtendedStates` struct, which first extends the reservoir states with training or prediction data,
then pads them with a specified value (defaulting to 1.0). This process is achieved through vertical concatenation,
combining the padding value, data, and states.
This function is particularly useful for enhancing the reservoir's state representation in more complex scenarios,
where both extended contextual information and consistent baseline padding are crucial for the computational
effectiveness of the reservoir computing model.
"""
function PaddedExtendedStates(; padding = 1.0)
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

#check matrix/vector
function (states_type::PaddedStates)(nla_type, x, y)
    tt = typeof(first(x))
    x_tmp = vcat(fill(tt(states_type.padding), (1, size(x, 2))), x)
    #x_tmp = reduce(vcat, x_tmp)
    return nla(nla_type, x_tmp)
end

#check matrix/vector
function (states_type::PaddedExtendedStates)(nla_type, x, y)
    tt = typeof(first(x))
    x_tmp = vcat(y, x)
    x_tmp = vcat(fill(tt(states_type.padding), (1, size(x, 2))), x_tmp)
    #x_tmp = reduce(vcat, x_tmp)
    return nla(nla_type, x_tmp)
end

#non linear algorithms
"""
    NLADefault()

`NLADefault` represents the default non-linear algorithm option. 
When used, it leaves the input array unchanged.
This option is suitable in cases where no non-linear transformation of the data is required,
maintaining the original state of the input array for further processing.
It's the go-to choice for preserving the raw data integrity within the computational pipeline
of the reservoir computing model.
"""
struct NLADefault <: NonLinearAlgorithm end

function nla(::NLADefault, x)
    return x
end

"""
    NLAT1()

`NLAT1` implements the T₁ transformation algorithm introduced in [^Chattopadhyay] and [^Pathak].
The T₁ algorithm selectively squares elements of the input array,
specifically targeting every second row. This non-linear transformation enhances certain data characteristics,
making it a valuable tool in analyzing chaotic systems and improving the performance of reservoir computing models.
The T₁ transformation's uniqueness lies in its selective approach, allowing for a more nuanced manipulation of the input data.

References:
[^Chattopadhyay]: Chattopadhyay, Ashesh, et al. 
    "Data-driven prediction of a multi-scale Lorenz 96 chaotic system using a
    hierarchy of deep learning methods: Reservoir computing, ANN, and RNN-LSTM." (2019).
[^Pathak]: Pathak, Jaideep, et al.
    "Model-free prediction of large spatiotemporally chaotic systems from data:
    A reservoir computing approach."
    Physical review letters 120.2 (2018): 024102.
"""
struct NLAT1 <: NonLinearAlgorithm end

function nla(::NLAT1, x_old)
    x_new = copy(x_old)
    for i in 1:size(x_new, 1)
        if mod(i, 2) != 0
            x_new[i, :] = copy(x_old[i, :] .* x_old[i, :])
        end
    end

    return x_new
end

"""
    NLAT2()

`NLAT2` implements the T₂ transformation algorithm as defined in [^Chattopadhyay].
This transformation algorithm modifies the reservoir states by multiplying each odd-indexed
row (starting from the second row) with the product of its two preceding rows.
This specific approach to non-linear transformation is useful for capturing and
enhancing complex patterns in the data, particularly beneficial in the analysis of chaotic
systems and in improving the dynamics within reservoir computing models.

Reference:
[^Chattopadhyay]: Chattopadhyay, Ashesh, et al.
    "Data-driven prediction of a multi-scale Lorenz 96 chaotic system using a
    hierarchy of deep learning methods: Reservoir computing, ANN, and RNN-LSTM." (2019).
"""
struct NLAT2 <: NonLinearAlgorithm end

function nla(::NLAT2, x_old)
    x_new = copy(x_old)
    for i in 2:(size(x_new, 1) - 1)
        if mod(i, 2) != 0
            x_new[i, :] = copy(x_old[i - 1, :] .* x_old[i - 2, :])
        end
    end

    return x_new
end

"""
    NLAT3()

The `NLAT3` struct implements the T₃ transformation algorithm as detailed in [^Chattopadhyay].
This algorithm modifies the reservoir's states by multiplying each odd-indexed row
(beginning from the second row) with the product of the immediately preceding and the
immediately following rows. T₃'s unique approach to data transformation makes it particularly
useful for enhancing complex data patterns, thereby improving the modeling and analysis
capabilities within reservoir computing, especially for chaotic and dynamic systems.

Reference:
[^Chattopadhyay]: Chattopadhyay, Ashesh, et al.
    "Data-driven prediction of a multi-scale Lorenz 96 chaotic system using a hierarchy of deep learning methods:
    Reservoir computing, ANN, and RNN-LSTM." (2019).
"""
struct NLAT3 <: NonLinearAlgorithm end

function nla(::NLAT3, x_old)
    x_new = copy(x_old)
    for i in 2:(size(x_new, 1) - 1)
        if mod(i, 2) != 0
            x_new[i, :] = copy(x_old[i - 1, :] .* x_old[i + 1, :])
        end
    end

    return x_new
end