module ReservoirComputing

using SparseArrays
using LinearAlgebra

include("init.jl")
export init_reservoir, init_input_layer
include("echostatenetwork.jl")
export states_matrix, esn_train, esn_predict

end #module
