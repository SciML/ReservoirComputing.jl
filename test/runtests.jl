using SafeTestsets
using Test

@testset "Common Utilities" begin
    @safetestset "Quality Assurance" begin
        include("qa.jl")
    end
    @safetestset "States" begin
        include("test_states.jl")
    end
end

@testset "Echo State Networks" begin
    @safetestset "ESN Input Layers" begin
        include("esn/test_input_layers.jl")
    end
    @safetestset "ESN Reservoirs" begin
        include("esn/test_reservoirs.jl")
    end
    @safetestset "ESN States" begin
        include("esn/test_states.jl")
    end
    @safetestset "ESN Train and Predict" begin
        include("esn/test_train.jl")
    end
    @safetestset "ESN Drivers" begin
        include("esn/test_drivers.jl")
    end
    @safetestset "Hybrid ESN" begin
        include("esn/test_hybrid.jl")
    end
end

@testset "CA based Reservoirs" begin
    @safetestset "RECA" begin
        include("reca/test_predictive.jl")
    end
end
