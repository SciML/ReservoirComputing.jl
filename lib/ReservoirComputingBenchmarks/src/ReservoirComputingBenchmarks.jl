module ReservoirComputingBenchmarks

import LinearAlgebra
using LinearAlgebra: I, cholesky, cholesky!, Symmetric, mul!, ldiv!, copytri!
using Statistics: mean, var, cor

const MetricFunction = Function

include("utils.jl")
include("memory_capacity.jl")
include("narma.jl")
include("ipc.jl")

export memory_capacity
export generate_narma, narma
export ipc
export nmse, rnmse, mse

end # module
