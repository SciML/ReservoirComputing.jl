using Test
using SafeTestsets
#using ReservoirComputing

@time @safetestset "Constructor" begin include("esn.jl") end
