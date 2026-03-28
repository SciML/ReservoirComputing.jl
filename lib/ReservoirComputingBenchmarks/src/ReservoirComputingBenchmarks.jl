module ReservoirComputingBenchmarks

import LinearAlgebra
using LinearAlgebra: I, cholesky, Symmetric, mul!
using Statistics: mean, var, cor

include("utils.jl")
include("memory_capacity.jl")
include("narma.jl")
include("ipc.jl")

export memory_capacity
export generate_narma, narma
export ipc

end # module
