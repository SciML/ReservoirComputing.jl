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
    @safetestset "Test initializers" begin
        include("esn/test_inits.jl")
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
    @safetestset "Deep ESN" begin
        include("esn/deepesn.jl")
    end
end

@testset "CA based Reservoirs" begin
    @safetestset "RECA" begin
        include("reca/test_predictive.jl")
    end
end
