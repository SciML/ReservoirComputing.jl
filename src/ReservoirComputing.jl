module ReservoirComputing

using SparseArrays
using LinearAlgebra
using MLJLinearModels
using LIBSVM
using GaussianProcesses
using Optim

abstract type AbstractEchoStateNetwork end
abstract type NonLinearAlgorithm end

include("ridge_train.jl")
export ESNtrain, Ridge, Lasso, ElastNet, RobustHuber

include("nla.jl")
export nla, NLADefault, NLAT1, NLAT2, NLAT3

include("esn_input_layers.jl") 
export init_input_layer, init_dense_input_layer, init_sparse_input_layer
include("esn_reservoirs.jl")
export init_reservoir_givendeg, init_reservoir_givensp

include("echostatenetwork.jl")
export ESN, ESNpredict, ESNpredict_h_steps

include("dafesn.jl")
export dafESN, dafESNpredict, dafESNpredict_h_steps

include("svesm.jl")
export SVESMtrain, SVESM_direct_predict

include("esgp.jl")
export ESGPtrain, ESGPpredict, ESGPpredict_h_steps
end #module
