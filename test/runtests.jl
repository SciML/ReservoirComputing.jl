using Test
using SafeTestsets
#using ReservoirComputing

@time @safetestset "esn" begin include("test_esn.jl") end
@time @safetestset "dafesn" begin include("test_dafesn.jl") end
