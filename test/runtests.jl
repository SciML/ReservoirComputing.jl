using SafeTestsets
using Test

@testset "Common Utilities" begin
    @safetestset "Quality Assurance" include("qa.jl")
    #@safetestset "States" include("test_states.jl")
end

@testset "Layers" begin
    @safetestset "Basic layers" include("layers/test_basic.jl")
    @safetestset "ESN Cell" include("layers/test_esncell.jl")
    @safetestset "SVMReadout" include("layers/test_svmreadout.jl")
end

@testset "Echo State Networks" begin
    @safetestset "ESN Initializers" include("esn/test_inits.jl")
    @safetestset "ESN model" include("esn/test_esn.jl")
    @safetestset "DeepESN model" include("esn/test_esn_deep.jl")
end
