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
using PartialFunctions
using Random
using SparseArrays
using Statistics
using WeightInitializers

export NLADefault, NLAT1, NLAT2, NLAT3
export StandardStates, ExtendedStates, PaddedStates, PaddedExtendedStates
export StandardRidge, LinearModel
export scaled_rand, weighted_init, sparse_init, informed_init, minimal_init
export rand_sparse, delay_line, delay_line_backward, cycle_jumps, simple_cycle, pseudo_svd
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

Given a set of labels as ```prediction_data```, this method of prediction will return the corresponding labels in a standard Machine Learning fashion.
"""
function Predictive(prediction_data)
    prediction_len = size(prediction_data, 2)
    Predictive(prediction_data, prediction_len)
end

#fallbacks for initializers
for initializer in (:rand_sparse, :delay_line, :delay_line_backward, :cycle_jumps,
    :simple_cycle, :pseudo_svd,
    :scaled_rand, :weighted_init, :sparse_init, :informed_init, :minimal_init)
    NType = ifelse(initializer === :rand_sparse, Real, Number)
    @eval function ($initializer)(dims::Integer...; kwargs...)
        return $initializer(_default_rng(), Float32, dims...; kwargs...)
    end
    @eval function ($initializer)(rng::AbstractRNG, dims::Integer...; kwargs...)
        return $initializer(rng, Float32, dims...; kwargs...)
    end
    @eval function ($initializer)(::Type{T},
            dims::Integer...; kwargs...) where {T <: $NType}
        return $initializer(_default_rng(), T, dims...; kwargs...)
    end
    @eval function ($initializer)(rng::AbstractRNG; kwargs...)
        return __partial_apply($initializer, (rng, (; kwargs...)))
    end
    @eval function ($initializer)(rng::AbstractRNG,
            ::Type{T}; kwargs...) where {T <: $NType}
        return __partial_apply($initializer, ((rng, T), (; kwargs...)))
    end
    @eval ($initializer)(; kwargs...) = __partial_apply($initializer, (; kwargs...))
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
include("esn/esn.jl")
include("esn/deepesn.jl")
include("esn/hybridesn.jl")
include("esn/esn_predict.jl")

#reca
include("reca/reca.jl")
include("reca/reca_input_encodings.jl")

end #module
