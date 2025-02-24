module ReservoirComputing

using Adapt: adapt
using CellularAutomata: CellularAutomaton, AbstractCA
using Compat: @compat
using LinearAlgebra: eigvals, mul!, I, qr, Diagonal
using NNlib: fast_act, sigmoid
using Random: Random, AbstractRNG, randperm, rand
using Reexport: Reexport, @reexport
using WeightInitializers: DeviceAgnostic, PartialFunction, Utils
@reexport using WeightInitializers

abstract type AbstractReservoirComputer end

@compat(public, (create_states))

#general
include("states.jl")
include("predict.jl")

#general training
include("train/linear_regression.jl")

#esn
include("esn/esn_inits.jl")
include("esn/esn_reservoir_drivers.jl")
include("esn/esn.jl")
include("esn/deepesn.jl")
include("esn/hybridesn.jl")
include("esn/esn_predict.jl")

#reca
include("reca/reca.jl")
include("reca/reca_input_encodings.jl")

export NLADefault, NLAT1, NLAT2, NLAT3
export StandardStates, ExtendedStates, PaddedStates, PaddedExtendedStates
export StandardRidge
export scaled_rand, weighted_init, informed_init, minimal_init, chebyshev_mapping,
       logistic_mapping, modified_lm
export rand_sparse, delay_line, delay_line_backward, cycle_jumps,
       simple_cycle, pseudo_svd, chaotic_init
export RNN, MRNN, GRU, GRUParams, FullyGated
export train
export ESN, HybridESN, KnowledgeModel, DeepESN
export RECA
export RandomMapping, RandomMaps
export Generative, Predictive, OutputLayer

end #module
