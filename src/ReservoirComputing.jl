module ReservoirComputing

using Adapt
using CellularAutomata
using Distances
using Distributions
using LIBSVM
using LinearAlgebra
using MLJLinearModels
using NNlib
using Optim
using SparseArrays
using Statistics

export NLADefault, NLAT1, NLAT2, NLAT3
export StandardStates, ExtendedStates, PaddedStates, PaddedExtendedStates
export StandardRidge, LinearModel
export AbstractLayer, create_layer
export WeightedLayer, DenseLayer, SparseLayer, MinimumLayer, InformedLayer, NullLayer
export BernoulliSample, IrrationalSample
export GaussianProcess
export AbstractReservoir, create_reservoir, create_states
export RandSparseReservoir, PseudoSVDReservoir, DelayLineReservoir
export DelayLineBackwardReservoir, SimpleCycleReservoir, CycleJumpsReservoir, NullReservoir
export RNN, MRNN, GRU, GRUParams, FullyGated, Variant1, Variant2, Variant3, Minimal
export ESN, Default, Hybrid, train
export RECA, train
export RandomMapping, RandomMaps
export Generative, Predictive, OutputLayer

#define global types
abstract type AbstractReservoirComputer end
abstract type AbstractOutputLayer end
abstract type AbstractPrediction end
#training methods
abstract type AbstractLinearModel end
abstract type AbstractSupportVector end
#should probably move some of these
abstract type AbstractVariation end
abstract type AbstractGRUVariant end

#general output layer struct
struct OutputLayer{T, I, S, L} <: AbstractOutputLayer
    training_method::T
    output_matrix::I
    out_size::S
    last_value::L
end

#prediction types
"""
    Generative(prediction_len)

This prediction methodology allows the models to produce an autonomous prediction, feeding the prediction into itself to generate the next step. 
The only parameter needed is the number of steps for the prediction.
"""
struct Generative{T} <: AbstractPrediction
    prediction_len::T
end

struct Predictive{I, T} <: AbstractPrediction
    prediction_data::I
    prediction_len::T
end

"""
    Predictive(prediction_data)

Given a set of labels as ```prediction_data``` this method of prediction will return the correspinding labels in a standard Machine Learning fashion.
"""
function Predictive(prediction_data)
    prediction_len = size(prediction_data, 2)
    Predictive(prediction_data, prediction_len)
end

#general
include("states.jl")
include("predict.jl")

#general training
include("train/linear_regression.jl")
include("train/supportvector_regression.jl")

#esn
include("esn/esn_input_layers.jl")
include("esn/esn_reservoirs.jl")
include("esn/esn_reservoir_drivers.jl")
include("esn/echostatenetwork.jl")
include("esn/esn_predict.jl")

#reca
include("reca/reca.jl")
include("reca/reca_input_encodings.jl")

end #module
