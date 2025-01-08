module ReservoirComputing

using Adapt: adapt
using CellularAutomata: CellularAutomaton
using LinearAlgebra: eigvals, mul!, I
using NNlib: fast_act, sigmoid
using Random: Random, AbstractRNG
using Reexport: Reexport, @reexport
using StatsBase: sample
using WeightInitializers: DeviceAgnostic, PartialFunction, Utils
@reexport using WeightInitializers

abstract type AbstractReservoirComputer end

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

export NLADefault, NLAT1, NLAT2, NLAT3
export StandardStates, ExtendedStates, PaddedStates, PaddedExtendedStates
export StandardRidge
export scaled_rand, weighted_init, informed_init, minimal_init
export rand_sparse, delay_line, delay_line_backward, cycle_jumps, simple_cycle, pseudo_svd
export RNN, MRNN, GRU, GRUParams, FullyGated, Minimal
export train
export ESN, HybridESN, KnowledgeModel, DeepESN
export RECA, sample
export RandomMapping, RandomMaps
export Generative, Predictive, OutputLayer

end #module
