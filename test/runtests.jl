using Pkg
using SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "All")

function activate_nopre_env()
    Pkg.activate("nopre")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core"
    @testset "Common Utilities" begin
        @safetestset "States" include("test_states.jl")
    end

    @testset "Layers" begin
        @safetestset "Basic layers" include("layers/test_basic.jl")
        @safetestset "ESN Cell" include("layers/test_esncell.jl")
        @safetestset "SVMReadout" include("layers/test_svmreadout.jl")
    end

    @testset "Echo State Networks" begin
        @safetestset "ESN Initializers" include("test_inits.jl")
        @safetestset "ESN model" include("models/test_esn.jl")
        @safetestset "DeepESN model" include("models/test_esn_deep.jl")
        @safetestset "DelayESN model" include("models/test_esn_delay.jl")
        @safetestset "HybridESN model" include("models/test_esn_hybrid.jl")
    end

    @testset "Next Generation Reservoir Computing" begin
        @safetestset "NGRC model" include("models/test_ngrc.jl")
    end
end

if GROUP == "nopre"
    activate_nopre_env()
    @safetestset "Quality Assurance" include("nopre/runtests.jl")
end
