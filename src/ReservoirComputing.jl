module ReservoirComputing

using Adapt
using CellularAutomata
using Distances
using LinearAlgebra
using NNlib
using Optim
using PartialFunctions
using Random
using Reexport: Reexport, @reexport
using Statistics
@reexport using WeightInitializers: WeightInitializers, DeviceAgnostic, PartialFunction,
                                    Utils, sparse_init

export NLADefault, NLAT1, NLAT2, NLAT3
export StandardStates, ExtendedStates, PaddedStates, PaddedExtendedStates
export StandardRidge
export scaled_rand, weighted_init, informed_init, minimal_init
export rand_sparse, delay_line, delay_line_backward, cycle_jumps, simple_cycle, pseudo_svd
export RNN, MRNN, GRU, GRUParams, FullyGated, Minimal
export train
export ESN
export HybridESN, KnowledgeModel
export DeepESN
export RECA
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

A prediction strategy that enables models to generate autonomous multi-step
forecasts by recursively feeding their own outputs back as inputs for
subsequent prediction steps.

# Parameters

  - `prediction_len::Int`: The number of future steps to predict.

# Description

The `Generative` prediction method allows a model to perform multi-step
forecasting by using its own previous predictions as inputs for future predictions.
This approach is especially useful in time series analysis, where each prediction
depends on the preceding data points.

At each step, the model takes the current input, generates a prediction,
and then incorporates that prediction into the input for the next step.
This recursive process continues until the specified
number of prediction steps (`prediction_len`) is reached.
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

A prediction strategy for supervised learning tasks,
where a model predicts labels based on a provided set
of input features (`prediction_data`).

# Parameters

  - `prediction_data`: The input data used for prediction, typically structured as a matrix
    where each column represents a sample, and each row represents a feature.

# Description

The `Predictive` prediction method is a standard approach
in supervised machine learning tasks. It uses the provided input data
(`prediction_data`) to produce corresponding labels or outputs based
on the learned relationships in the model. Unlike generative prediction,
this method does not recursively feed predictions into the model;
instead, it operates on fixed input data to produce a single batch of predictions.

This method is suitable for tasks like classification,
regression, or other use cases where the input features
and the number of steps are predefined.
"""
function Predictive(prediction_data)
    prediction_len = size(prediction_data, 2)
    Predictive(prediction_data, prediction_len)
end

__partial_apply(fn, inp) = fn$inp

#fallbacks for initializers #eventually to remove once migrated to WeightInitializers.jl
for initializer in (:rand_sparse, :delay_line, :delay_line_backward, :cycle_jumps,
    :simple_cycle, :pseudo_svd,
    :scaled_rand, :weighted_init, :informed_init, :minimal_init)
    @eval begin
        function ($initializer)(dims::Integer...; kwargs...)
            return $initializer(Utils.default_rng(), Float32, dims...; kwargs...)
        end
        function ($initializer)(rng::AbstractRNG, dims::Integer...; kwargs...)
            return $initializer(rng, Float32, dims...; kwargs...)
        end
        function ($initializer)(::Type{T}, dims::Integer...; kwargs...) where {T <: Number}
            return $initializer(Utils.default_rng(), T, dims...; kwargs...)
        end

        # Partial application
        function ($initializer)(rng::AbstractRNG; kwargs...)
            return PartialFunction.Partial{Nothing}($initializer, rng, kwargs)
        end
        function ($initializer)(::Type{T}; kwargs...) where {T <: Number}
            return PartialFunction.Partial{T}($initializer, nothing, kwargs)
        end
        function ($initializer)(rng::AbstractRNG, ::Type{T}; kwargs...) where {T <: Number}
            return PartialFunction.Partial{T}($initializer, rng, kwargs)
        end
        function ($initializer)(; kwargs...)
            return PartialFunction.Partial{Nothing}($initializer, nothing, kwargs)
        end
    end
end

#general
include("states.jl")
include("predict.jl")

#general training
include("train/linear_regression.jl")

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

# Julia < 1.9 support 
if !isdefined(Base, :get_extension)
    include("../ext/RCMLJLinearModelsExt.jl")
    include("../ext/RCLIBSVMExt.jl")
end

end #module
