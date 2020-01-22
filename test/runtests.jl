using Test

@testset "ReservoirComputing" begin
    tests = [
        "esn_test",
    ]
    res = map(tests) do t
        @eval module $(Symbol("Test_", t))
            include($t * ".jl")
        end
        return
    end
end
