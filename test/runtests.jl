using SafeTestsets
using Test

@testset "Common Utilities" begin
    @safetestset "Quality Assurance" include("qa.jl")
    @safetestset "States" include("test_states.jl")
end

@testset "Echo State Networks" begin
    @safetestset "ESN Input Layers" include("esn/test_input_layers.jl")
    @safetestset "ESN Reservoirs" include("esn/test_reservoirs.jl")
    @safetestset "ESN States" include("esn/test_states.jl")
    @safetestset "ESN Train and Predict" include("esn/test_train.jl")
    @safetestset "ESN Drivers" include("esn/test_drivers.jl")
    @safetestset "Hybrid ESN" include("esn/test_hybrid.jl")
end

@testset "CA based Reservoirs" begin
    @safetestset "RECA" include("reca/test_predictive.jl")
end
