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
abstract type AbstractEchoStateNetwork <: AbstractReservoirComputer end
abstract type NonLinearAlgorithm end

include("esn/leaky_fixed_rnn.jl")
include("esn/train.jl")
export ESNtrain, Ridge, Lasso, ElastNet, RobustHuber, HESNtrain

include("esn/nla.jl")
export nla, NLADefault, NLAT1, NLAT2, NLAT3

include("esn/esn_input_layers.jl")
export init_input_layer, init_dense_input_layer, init_sparse_input_layer, min_complex_input, irrational_sign_input, physics_informed_input
include("esn/esn_reservoirs.jl")
export init_reservoir_givendeg, init_reservoir_givensp, pseudoSVD, DLR, DLRB, SCR, CRJ

include("esn/echostatenetwork.jl")
export ESN, ESNpredict, ESNpredict_h_steps, ESNfitted

include("esn/dafesn.jl")
export dafESN, dafESNpredict, dafESNpredict_h_steps

include("esn/svesm.jl")
export SVESMtrain, SVESM_direct_predict, SVESMpredict, SVESMpredict_h_steps

include("esn/esgp.jl")
export ESGPtrain, ESGPpredict, ESGPpredict_h_steps

include("esn/gruesn.jl")
export GRUESN, GRUESNpredict

include("esn/hesn.jl")
export HESN, HESNpredict

include("reca/reca_discrete.jl")
export RECA_discrete, RECAdirect_predict_discrete

include("reca/reca_gol.jl")
export RECA_TwoDim, RECATDdirect_predict_discrete, RECATD_predict_discrete



end #module
