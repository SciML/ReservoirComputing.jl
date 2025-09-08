using SafeTestsets
using Test

@testset "Common Utilities" begin
    @safetestset "Quality Assurance" include("qa.jl")
    #@safetestset "States" include("test_states.jl")
end

#@testset "Echo State Networks" begin
#    @safetestset "ESN Initializers" include("esn/test_inits.jl")
#end
