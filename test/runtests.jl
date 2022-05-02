using Test
using SafeTestsets

@testset "Commons" begin
    @safetestset "States" begin include("test_states.jl") end
end

@testset "Echo State Network tests" begin
    @safetestset "ESN Input Layers" begin include("esn/test_input_layers.jl") end
    @safetestset "ESN Reservoirs" begin include("esn/test_reservoirs.jl") end
    @safetestset "ESN States" begin include("esn/test_states.jl") end
    @safetestset "ESN Train and Predict" begin include("esn/test_train.jl") end
    @safetestset "ESN Drivers" begin include("esn/test_drivers.jl") end
    @safetestset "ESN Non Linear Algos" begin include("esn/test_nla.jl") end
end

@testset "Reservoir Computing with Cellular Automata tests" begin
    @safetestset "RECA" begin include("reca/test_predictive.jl") end
end
