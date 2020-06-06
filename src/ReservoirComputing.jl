module ReservoirComputing

using SparseArrays
using LinearAlgebra
using MLJLinearModels

abstract type AbstractEchoStateNetwork end
abstract type NonLinearAlgorithm end

include("ridge_train.jl")
export ESNtrain, Ridge, Lasso, ElastNet, RobustHuber

include("nla.jl")
export nla, NLADefault, NLAT1, NLAT2, NLAT3

include("esn_input_layers.jl") 
include("esn_reservoirs.jl")

include("echostatenetwork.jl")
export ESN, ESNpredict, ESNpredict_h_steps

include("dafesn.jl")
export dafESN, dafESNpredict


end #module
