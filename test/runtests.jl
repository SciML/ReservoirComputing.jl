using Test
using SafeTestsets


@time @safetestset "ESN constructors" begin include("constructors/test_esn_constructors.jl") end
@time @safetestset "HESN constructors" begin include("constructors/test_hesn_constructors.jl") end
@time @safetestset "DAFESN constructors" begin include("constructors/test_dafesn_constructors.jl") end
@time @safetestset "ESN input layers" begin include("fixed_layers/test_esn_input_layers.jl") end
@time @safetestset "ESN reservoirs" begin include("fixed_layers/test_esn_reservoirs.jl") end
@time @safetestset "Non Linear Algorithms" begin include("extras/test_nla.jl") end
@time @safetestset "Extended states" begin include("extras/test_extended_states.jl") end
@time @safetestset "MLJ Linear Models for ESN" begin include("training/test_mlj_lm.jl") end
@time @safetestset "ESGP" begin include("training/test_esgp.jl") end
@time @safetestset "RECA_discrete constructors" begin include("constructors/test_recadiscrete_constructors.jl") end
@time @safetestset "RECA gol constructors" begin include("constructors/test_recagol_constructors.jl") end
@time @safetestset "reca discrete predict" begin include("training/test_recadiscrete.jl") end
@time @safetestset "reca gol predict" begin include("training/test_recagol.jl") end
@time @safetestset "RMM constructors" begin include("constructors/test_rmm_constructors.jl") end
@time @safetestset "GRUESN constructors" begin include("constructors/test_gruesn_constructors.jl") end
