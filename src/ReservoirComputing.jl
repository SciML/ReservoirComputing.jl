module ReservoirComputing

using SparseArrays
using LinearAlgebra

include("echostatenetwork.jl")
export ESN, ESNtrain, ESNpredict, ESNsingle_predict

end #module
