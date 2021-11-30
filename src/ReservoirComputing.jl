module ReservoirComputing

using SparseArrays
using LinearAlgebra
using MLJLinearModels
using LIBSVM
using GaussianProcesses
using Optim
using Distributions
using Statistics
using Distances
using NNlib
using CellularAutomata

#define global types
abstract type AbstractReservoirComputer end
abstract type AbstractPrediction end
abstract type NonLinearAlgorithm end
abstract type AbstractStates end
#should probably move some of these
abstract type AbstractVariation end
abstract type AbstractLayer end
abstract type AbstractReservoirDriver end
abstract type AbstractReservoir end
abstract type AbstractOutputLayer end
abstract type AbstractLinearModel end
abstract type AbstractGaussianProcess end
abstract type AbstractSupportVector end
abstract type AbstractGRUVariant end


#general output layer struct
struct OutputLayer{T,I,S,L} <: AbstractOutputLayer
    training_method::T
    output_matrix::I
    out_size::S
    last_value::L
end

#prediction types
struct Generative{T} <: AbstractPrediction
    prediction_len::T
end

struct Predictive{I,T} <: AbstractPrediction
    prediction_data::I
    prediction_len::T
end

function Predictive(prediction_data)
    prediction_len = size(prediction_data, 2)
    Predictive(prediction_data, prediction_len)
end

#states types
struct ExtendedStates <: AbstractStates end
struct StandardStates <: AbstractStates end
struct PaddedStates{T} <: AbstractStates
    padding::T
end

struct PaddedExtendedStates{T} <: AbstractStates 
    padding::T
end

function PaddedStates(;padding=1.0)
    PaddedStates(padding)
end

function PaddedExtendedStates(;padding=1.0)
    PaddedExtendedStates(padding)
end

function (states_type::ExtendedStates)(nla_type, x, y)
    x_tmp = vcat(y, x)
    nla(nla_type, x_tmp)
end

function (states_type::StandardStates)(nla_type, x, y)
    nla(nla_type, x)
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

#import/export
#general
include("nla.jl")
export nla, NLADefault, NLAT1, NLAT2, NLAT3
include("predict.jl")
export obtain_prediction

#general training
include("train/linear_regression.jl")
export _train, StandardRidge, LinearModel
include("train/gaussian_regression.jl")
export _train, GaussianProcess
include("train/supportvector_regression.jl")
export _train

#esn
include("esn/echostatenetwork.jl")
export ESN, Standard, Hybrid, next_state_prediction, train
include("esn/esn_input_layers.jl")
export create_layer, WeightedLayer, DenseLayer, SparseLayer, MinimumLayer, InformedLayer
BernoulliSample, IrrationalSample, NullLayer
include("esn/esn_reservoir_drivers.jl")
export next_state, create_states, RNN, GRU, GRUParams, Variant1, Variant2, Variant3, Minimal
include("esn/esn_reservoirs.jl")
export create_reservoir, RandSparseReservoir, PseudoSVDReservoir, DelayLineReservoir,
DelayLineBackwardReservoir, SimpleCycleReservoir, CycleJumpsReservoir, NullReservoir


#reca
include("reca/reca.jl")
export RECA, train, next_state_prediction
include("reca/reca_input_encodings.jl")
export RandomMapping, RandomMaps



export Generative, Predictive, OutputLayer, states_type,
StandardStates, ExtendedStates, PaddedStates, PaddedExtendedStates


end #module
