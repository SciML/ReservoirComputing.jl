module ReservoirComputingBenchmarks

import LinearAlgebra
using LinearAlgebra: I, cholesky, cholesky!, Symmetric, mul!, ldiv!, copytri!
using Statistics: mean, var, cor

const MetricFunction = Function

include("utils.jl")
include("memory_capacity.jl")
include("nonlinear_memory.jl")
include("nonlinear_transformation.jl")
include("sin_approximation.jl")
include("narma.jl")
include("ipc.jl")
include("kernel_rank.jl")

export memory_capacity, nonlinear_memory
export nonlinear_transformation, sin_approximation
export generate_narma, narma
export ipc
export kernel_rank, generalization_rank
export nmse, rnmse, mse

end # module
