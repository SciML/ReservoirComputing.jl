using Test
using SafeTestsets

@testset "Echo State Network tests" begin
    @safetestset "ESN Input Layers" begin include("esn/test_input_layers.jl") end
    @safetestset "ESN Reservoirs" begin include("esn/test_reservoirs.jl") end
    @safetestset "ESN States" begin include("esn/test_states.jl") end
    @safetestset "ESN Train and Predict" begin include("esn/test_train.jl") end
    @safetestset "ESN GRU" begin include("esn/test_gru.jl") end

end
