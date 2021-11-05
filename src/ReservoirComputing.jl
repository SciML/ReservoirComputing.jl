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

#define global types
abstract type AbstractReservoirComputer end
abstract type AbstractPrediction end
abstract type NonLinearAlgorithm end
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
struct OutputLayer{T,I,S} <: AbstractOutputLayer
    training_method::T
    output_matrix::I
    out_size::S
end

#prediction types
struct Autonomous{T} <: AbstractPrediction
    prediction_len::T
end

struct Direct{T} <: AbstractPrediction
    prediction_data::T
end

function Direct(;prediction_data=nothing)
    Direct(prediction_data)
end

struct Fitted{T} <: AbstractPrediction
    type::T #Autonomous or Direct
end

function Fitted(;type=Direct())
    Fitted(type)
end


#import/export
include("nla.jl")
export nla, NLADefault, NLAT1, NLAT2, NLAT3

include("esn/echostatenetwork.jl")
export ESN, Standard, Hybrid
include("esn/esn_input_layers.jl")
export create_layer, WeightedLayer, DenseLayer, SparseLayer, MinimumLayer, InformedLayer
BernoulliSample, IrrationalSample
include("esn/esn_reservoir_drivers.jl")
export next_state, create_states, RNN, GRU, GRUParams, Variant1, Variant2, Variant3, Minimal
include("esn/esn_reservoirs.jl")
export create_reservoir, RandSparseReservoir, PseudoSVDReservoir, DelayLineReservoir,
DelayLineBackwardReservoir, SimpleCycleReservoir, CycleJumpsReservoir
include("esn/esn_train.jl")
export train
include("esn/esn_predict.jl")
export obtain_autonomous_prediction, obtain_direct_prediction

include("train/linear_regression.jl")
export _train, StandardRidge, LinearModel
include("train/gaussian_regression.jl")
export _train, GaussianProcess
include("train/supportvector_regression.jl")
export _train

export Autonomous, Direct, OutputLayer, Fitted


end #module
