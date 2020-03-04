module ReservoirComputing

using SparseArrays
using LinearAlgebra

include("echostatenetwork.jl")
export ESN, ESNtrain, ESNpredict, ESNsingle_predict

include("dafesn.jl")
export dafESN, dafESNtrain, dafESNpredict


end #module
