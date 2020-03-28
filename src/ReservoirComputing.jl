module ReservoirComputing

using SparseArrays
using LinearAlgebra

include("nonlinalg.jl")
export NonLinAlgDefault, NonLinAlgT1, NonLinAlgT2, NonLinAlgT3
include("echostatenetwork.jl")
export ESN, ESNtrain, ESNpredict, ESNsingle_predict

include("dafesn.jl")
export dafESN, dafESNtrain, dafESNpredict


end #module
