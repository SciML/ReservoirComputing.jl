module ReservoirComputing

using Adapt: adapt
using Compat: @compat
using LinearAlgebra: eigvals, mul!, I, qr, Diagonal
using NNlib: fast_act, sigmoid
using Random: Random, AbstractRNG, randperm
using Reexport: Reexport, @reexport
using WeightInitializers: DeviceAgnostic, PartialFunction, Utils
@reexport using WeightInitializers

abstract type AbstractReservoirComputer end

@compat(public, (create_states))

#general
include("generics/states.jl")
include("generics/predict.jl")
include("generics/linear_regression.jl")
#extensions
include("extensions/reca.jl")
#esn
include("inits/inits_components.jl")
include("inits/esn_inits.jl")
include("layers/esn_reservoir_drivers.jl")
include("models/esn.jl")
include("models/deepesn.jl")
include("models/hybridesn.jl")
include("models/esn_predict.jl")

export NLADefault, NLAT1, NLAT2, NLAT3, PartialSquare, ExtendedSquare
export StandardStates, ExtendedStates, PaddedStates, PaddedExtendedStates
export StandardRidge
export chebyshev_mapping, informed_init, logistic_mapping, minimal_init,
       modified_lm, scaled_rand, weighted_init, weighted_minimal
export block_diagonal, chaotic_init, cycle_jumps, delay_line, delay_line_backward,
       double_cycle, forward_connection, low_connectivity, pseudo_svd, rand_sparse,
       selfloop_cycle, selfloop_delayline_backward, selfloop_feedback_cycle,
       selfloop_forward_connection, simple_cycle, true_double_cycle
export add_jumps!, backward_connection!, delay_line!, reverse_simple_cycle!,
       scale_radius!, self_loop!, simple_cycle!
export RNN, MRNN, GRU, GRUParams, FullyGated, Minimal
export train
export ESN, HybridESN, KnowledgeModel, DeepESN
export Generative, Predictive, OutputLayer
#reca
export RECA
export RandomMapping, RandomMaps

end #module
