module ReservoirComputing

using Adapt
using CellularAutomata
using Distances
using Distributions
using LinearAlgebra
using NNlib
using Optim
using PartialFunctions
using Random
using Statistics
using WeightInitializers

export NLADefault, NLAT1, NLAT2, NLAT3
export StandardStates, ExtendedStates, PaddedStates, PaddedExtendedStates
export StandardRidge
export RNN, MRNN, GRU, GRUParams, FullyGated, Minimal
export ESN, train
export HybridESN, KnowledgeModel
export DeepESN
export RECA, train
export RandomMapping, RandomMaps
export Generative, Predictive, OutputLayer

#define global types
abstract type AbstractReservoirComputer end
abstract type AbstractOutputLayer end
abstract type AbstractPrediction end
#should probably move some of these
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

Given a set of labels as `prediction_data`, this method of prediction will return the corresponding labels in a standard Machine Learning fashion.
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

#esn
include("esn/esn_reservoir_drivers.jl")
include("esn/esn.jl")
include("esn/deepesn.jl")
include("esn/hybridesn.jl")
include("esn/esn_predict.jl")

#reca
include("reca/reca.jl")
include("reca/reca_input_encodings.jl")

# Julia < 1.9 support 
if !isdefined(Base, :get_extension)
    include("../ext/RCMLJLinearModelsExt.jl")
    include("../ext/RCLIBSVMExt.jl")
end

end #module
