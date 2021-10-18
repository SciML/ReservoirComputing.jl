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


abstract type AbstractReservoirComputer end
abstract type AbstractPrediction end
abstract type NonLinearAlgorithm end
abstract type AbstractInputLayer end
abstract type AbstractReservoirDriver end
abstract type AbstractReservoir end
abstract type LinearModel end



include("esn/echostatenetwork.jl")
export ESN, Autonomous, Direct
include("esn/nla.jl")
export nla, NLADefault, NLAT1, NLAT2, NLAT3
include("esn/esn_input_layers.jl")
export create_input_layer, WeightedInput, DenseInput, SparseInput, MinimumInput
include("esn/esn_reservoir_drivers.jl")
export next_state, create_states, RNN
include("esn/esn_reservoirs.jl")
export create_reservoir, RandSparseReservoir, PseudoSVDReservoir, DelayLineReservoir,
DelayLineBackwardReservoir, SimpleCycleReservoir, CycleJumpsReservoir

include("train/linear_regression.jl")
export train!, StandardRidge, Ridge, Lasso, ElastNet, RobustHuber



end #module
