using ReservoirComputing, BenchmarkTools
using LuxCore, StableRNGs, Random

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

in_dims, res_dims, out_dims = 2, 100, 1
n_steps = 100
train_data = randn(rng, Float32, in_dims, n_steps)
target_data = randn(rng, Float32, out_dims, n_steps)
input_step = randn(rng, Float32, in_dims)

# =============================================================================
# Reservoir computer construction and inference
# =============================================================================

SUITE["esn"] = BenchmarkGroup()

SUITE["esn"]["construct"] = @benchmarkable ESN(
    $in_dims, $res_dims, $out_dims
)
SUITE["esn"]["construct_sparse"] = @benchmarkable ESN(
    $in_dims, $res_dims, $out_dims; init_reservoir = sparse_init
)
SUITE["esn"]["construct_cycle"] = @benchmarkable ESN(
    $in_dims, $res_dims, $out_dims; init_reservoir = cycle_jumps
)

esn = ESN(in_dims, res_dims, out_dims)
ps, st = setup(rng, esn)

SUITE["esn"]["setup"] = @benchmarkable setup($rng, $esn)
SUITE["esn"]["step"] = @benchmarkable $esn($input_step, $ps, $st)

# =============================================================================
# Training (ridge-regression readout)
# =============================================================================

SUITE["train"] = BenchmarkGroup()

ridge = RidgeRegression(2.0e-3)
SUITE["train"]["train"] = @benchmarkable train(
    $esn, $train_data, $target_data, $ps, $st; objective = $ridge
)

# =============================================================================
# Reservoir inits
# =============================================================================

SUITE["init"] = BenchmarkGroup()

SUITE["init"]["rand_sparse"] = @benchmarkable sparse_init(
    $rng, Float32, $res_dims, $res_dims; sparsity = 0.1
)
SUITE["init"]["delay_line"] = @benchmarkable delay_line(
    $rng, Float32, $res_dims, $res_dims
)
SUITE["init"]["chaotic"] = @benchmarkable chaotic_init(
    $rng, Float32, $res_dims, $res_dims
)
