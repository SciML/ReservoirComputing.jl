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
using Adapt


#define global types
abstract type AbstractReservoirComputer end
abstract type AbstractOutputLayer end
abstract type AbstractPrediction end
#training methods
abstract type AbstractLinearModel end
abstract type AbstractGaussianProcess end
abstract type AbstractSupportVector end
#should probably move some of these
abstract type AbstractVariation end
abstract type AbstractGRUVariant end


#general output layer struct
struct OutputLayer{T,I,S,L} <: AbstractOutputLayer
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

struct Predictive{I,T} <: AbstractPrediction
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



#import/export
#general
include("states.jl")
export nla, NLADefault, NLAT1, NLAT2, NLAT3,
StandardStates, ExtendedStates, PaddedStates, PaddedExtendedStates
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
include("esn/esn_input_layers.jl")
export AbstractLayer, create_layer, WeightedLayer, DenseLayer, SparseLayer, MinimumLayer, InformedLayer, NullLayer,
BernoulliSample, IrrationalSample
include("esn/esn_reservoirs.jl")
export AbstractReservoir, create_reservoir, RandSparseReservoir, PseudoSVDReservoir, DelayLineReservoir,
DelayLineBackwardReservoir, SimpleCycleReservoir, CycleJumpsReservoir, NullReservoir
include("esn/esn_reservoir_drivers.jl")
export next_state, create_states, RNN, MRNN, GRU, GRUParams, FullyGated, Variant1, Variant2, Variant3, Minimal
include("esn/echostatenetwork.jl")
export ESN, Default, Hybrid, next_state_prediction, train
include("esn/esn_predict.jl")

#reca
include("reca/reca.jl")
export RECA, train, next_state_prediction
include("reca/reca_input_encodings.jl")
export RandomMapping, RandomMaps



export Generative, Predictive, OutputLayer


end #module
