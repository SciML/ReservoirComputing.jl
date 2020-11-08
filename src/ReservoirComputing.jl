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

include("leaky_fixed_rnn.jl")
include("train.jl")
export ESNtrain, Ridge, Lasso, ElastNet, RobustHuber, HESNtrain

include("nla.jl")
export nla, NLADefault, NLAT1, NLAT2, NLAT3

include("esn_input_layers.jl")
export init_input_layer, init_dense_input_layer, init_sparse_input_layer, min_complex_input, irrational_sign_input, physics_informed_input
include("esn_reservoirs.jl")
export init_reservoir_givendeg, init_reservoir_givensp, pseudoSVD, DLR, DLRB, SCR, CRJ

include("echostatenetwork.jl")
export ESN, ESNpredict, ESNpredict_h_steps

include("dafesn.jl")
export dafESN, dafESNpredict, dafESNpredict_h_steps

include("svesm.jl")
export SVESMtrain, SVESM_direct_predict, SVESMpredict, SVESMpredict_h_steps

include("esgp.jl")
export ESGPtrain, ESGPpredict, ESGPpredict_h_steps

include("ECA.jl")
export ECA
#include("reca.jl")
#export RECA, reca_predict
include("reca_discrete.jl")
export RECA_discrete, RECAdirect_predict_discrete
include("gameoflife.jl")
export GameOfLife
include("reca_gol.jl")
export RECA_TwoDim, RECATDdirect_predict_discrete, RECATD_predict_discrete

include("rmm.jl")
export RMM, RMMdirect_predict

include("gruesn.jl")
export GRUESN, GRUESNpredict

include("hesn.jl")
export HESN, HESNpredict

end #module
