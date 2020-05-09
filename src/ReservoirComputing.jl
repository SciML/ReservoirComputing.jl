module ReservoirComputing

using SparseArrays
using LinearAlgebra

abstract type AbstractEchoStateNetwork end
abstract type NonLinearAlgorithm end


include("nla.jl")
export nla, NLADefault, NLAT1, NLAT2, NLAT3
include("nonlinalg.jl")
export NonLinAlgDefault, NonLinAlgT1, NonLinAlgT2, NonLinAlgT3
include("echostatenetwork.jl")
export ESN, ESNtrain, ESNpredict, ESNsingle_predict

include("dafesn.jl")
export dafESN, dafESNtrain, dafESNpredict


end #module
