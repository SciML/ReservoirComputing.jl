using SafeTestsets
using Test

@testset "Common Utilities" begin
    @safetestset "Quality Assurance" include("qa.jl")
end

@testset "CA based Reservoirs" begin
    @safetestset "RECA" include("test_predictive.jl")
end
