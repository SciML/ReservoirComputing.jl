using Test
using SafeTestsets

@testset "Echo State Network tests" begin
    @safetestset "ESN Input Layers" begin include("esn/test_input_layers.jl") end
    @safetestset "ESN Reservoirs" begin include("esn/test_reservoirs.jl") end
    @safetestset "ESN States" begin include("esn/test_esn_states.jl") end
end
