using Test
using Random
using LinearAlgebra
using Statistics
using ReservoirComputing
using OrdinaryDiffEq
using SciMLBase
using DataInterpolations

# ---------------------------------------------------------------------------
# 1. Construction: shape + parameter wiring
#
# `ContinuousESN(in_dims, res_dims, tspan, solver)` should build a reservoir
# whose `initialparameters` yields the canonical eq-(5) trio (`W_r`,
# `W_in`, `inv_leak`) at the correct shapes, with no bias by default.
# ---------------------------------------------------------------------------

@testset "ContinuousESN: construction + parameter shapes" begin
    rng = MersenneTwister(0)
    in_dim, res_dim = 3, 50
    ce = ContinuousESN(in_dim, res_dim, (0.0, 5.0), Tsit5())

    rc = ReservoirComputer(ce, LinearReadout(res_dim => 2))
    ps, st = setup(rng, rc)

    @test ce isa AbstractSciMLProblemReservoir
    @test size(ps.reservoir.W_r) == (res_dim, res_dim)
    @test size(ps.reservoir.W_in) == (res_dim, in_dim)
    @test ps.reservoir.leak_coefficient == 1.0f0
    @test !haskey(ps.reservoir, :b)
    @test all(isfinite, ps.reservoir.W_r)
    @test all(isfinite, ps.reservoir.W_in)
end

# ---------------------------------------------------------------------------
# 2. Construction validation: argument errors
#
# Positive `in_dims`, `res_dims`, `leak_coefficient`, and strictly
# increasing `tspan` are required. Protected `solve` kwargs are rejected
# at construction (inherits `SciMLProblemReservoir`'s _PROTECTED_SOLVE_KWARGS).
# ---------------------------------------------------------------------------

@testset "ContinuousESN: construction validation" begin
    @test_throws ArgumentError ContinuousESN(0, 5, (0.0, 1.0), Tsit5())
    @test_throws ArgumentError ContinuousESN(3, 0, (0.0, 1.0), Tsit5())
    @test_throws ArgumentError ContinuousESN(3, 5, (0.0, 0.0), Tsit5())
    @test_throws ArgumentError ContinuousESN(3, 5, (1.0, 0.0), Tsit5())
    @test_throws ArgumentError ContinuousESN(
        3, 5, (0.0, 1.0), Tsit5(); leak_coefficient = 0.0
    )
    # Non-finite endpoints
    @test_throws ArgumentError ContinuousESN(3, 5, (0.0, Inf), Tsit5())
    @test_throws ArgumentError ContinuousESN(3, 5, (-Inf, 1.0), Tsit5())
    @test_throws ArgumentError ContinuousESN(3, 5, (0.0, NaN), Tsit5())
    # Wrong-length tspan
    @test_throws ArgumentError ContinuousESN(3, 5, (1.0,), Tsit5())
    @test_throws ArgumentError ContinuousESN(3, 5, (0.0, 1.0, 2.0), Tsit5())
    # leak_coefficient that overflows `T` is caught at `setup` time, not
    # construction (the construction check sees the unconverted value).
    # Use a dense init to avoid `rand_sparse`'s NaN-on-tiny-matrix edge.
    let dense_init = (rng, T, d...) -> rand_sparse(rng, T, d...; sparsity = 0.5)
        ce_overflow = ContinuousESN(
            3, 16, (0.0, 1.0), Tsit5();
            leak_coefficient = 1.0e50, T = Float32, init_reservoir = dense_init
        )
        rc_overflow = ReservoirComputer(ce_overflow, LinearReadout(16 => 1))
        @test_throws ArgumentError setup(MersenneTwister(0), rc_overflow)
    end
    for badkw in (:saveat, :save_everystep, :dense)
        @test_throws ArgumentError ContinuousESN(
            3, 5, (0.0, 1.0), Tsit5(); (badkw => true,)...
        )
    end
end

# ---------------------------------------------------------------------------
# 3. Bias toggle
#
# `use_bias = true` should add a `b` parameter of length `res_dims`.
# ---------------------------------------------------------------------------

@testset "ContinuousESN: bias toggle" begin
    rng = MersenneTwister(1)
    in_dim, res_dim = 2, 10
    ce_b = ContinuousESN(in_dim, res_dim, (0.0, 1.0), Tsit5(); use_bias = true)
    rc = ReservoirComputer(ce_b, LinearReadout(res_dim => 1))
    ps, st = setup(rng, rc)
    @test haskey(ps.reservoir, :b)
    @test length(ps.reservoir.b) == res_dim
    # Default init is the type-aware zeros initialiser.
    @test all(==(0), ps.reservoir.b)
end

# ---------------------------------------------------------------------------
# 3b. Bias actually exercised under an adaptive solver
#
# The `haskey(:b)` branch in `_continuous_esn_rhs!` is solver-agnostic in
# theory, but only the Euler-equivalence test (which uses `Euler`) hits
# the bias-on path under integration. Pair `use_bias = true` with a
# nonzero bias initialiser and `Tsit5` to confirm: (a) the bias does
# perturb states relative to the no-bias baseline, and (b) the result
# stays finite.
# ---------------------------------------------------------------------------

@testset "ContinuousESN: bias path under Tsit5" begin
    rng = MersenneTwister(101)
    in_dim, res_dim, T_steps = 2, 20, 12
    nonzero_bias(rng, T, d...) = T(0.5) .* randn(rng, T, d...)
    ce_with_bias = ContinuousESN(
        in_dim, res_dim, (0.0, 2.0), Tsit5();
        use_bias = true, init_bias = nonzero_bias,
        reltol = 1.0e-8, abstol = 1.0e-10
    )
    ce_no_bias = ContinuousESN(
        in_dim, res_dim, (0.0, 2.0), Tsit5();
        use_bias = false, reltol = 1.0e-8, abstol = 1.0e-10
    )
    rc_b = ReservoirComputer(ce_with_bias, LinearReadout(res_dim => 1))
    rc_n = ReservoirComputer(ce_no_bias, LinearReadout(res_dim => 1))
    ps_b, st_b = setup(MersenneTwister(0), rc_b)
    ps_n, st_n = setup(MersenneTwister(0), rc_n)
    data = randn(Float32, in_dim, T_steps)

    s_b, _ = collectstates(rc_b, data, ps_b, st_b)
    s_n, _ = collectstates(rc_n, data, ps_n, st_n)
    @test all(isfinite, s_b)
    @test s_b != s_n
end

# ---------------------------------------------------------------------------
# 4. Forward pass: shape + finiteness + determinism
# ---------------------------------------------------------------------------

@testset "ContinuousESN: forward (collectstates)" begin
    rng = MersenneTwister(7)
    in_dim, res_dim, T_steps = 2, 16, 25
    ce = ContinuousESN(
        in_dim, res_dim, (0.0, 3.0), Tsit5();
        reltol = 1.0e-8, abstol = 1.0e-10
    )
    rc = ReservoirComputer(ce, LinearReadout(res_dim => 1))
    ps, st = setup(rng, rc)
    data = randn(Float32, in_dim, T_steps)

    s1, _ = collectstates(rc, data, ps, st)
    s2, _ = collectstates(rc, data, ps, st)
    @test size(s1) == (res_dim, T_steps)
    @test all(isfinite, s1)
    @test s1 ≈ s2
end

# ---------------------------------------------------------------------------
# 5. Euler equivalence with discrete leaky ESN
#
# Solving `ẋ = α(-x + tanh(W_r·x + W_in·u(t) + b))` with explicit Euler at
# step `Δt = 1` and `leak_coefficient = α` collapses algebraically to the
# discrete leaky ESN update `x_{k+1} = (1-α)·x_k + α·tanh(...)`. With the
# corrected window alignment (input at window start, sample at window
# end), continuous and discrete trajectories agree exactly.
#
# Covers both vanilla (α = 1) and leaky (α = 0.3) regimes.
# ---------------------------------------------------------------------------

@testset "ContinuousESN: Euler equivalence with discrete leaky ESN" begin
    # Dense reservoir init avoids `rand_sparse`'s spectral-scaling edge case
    # on very small matrices (8×8 at 10% density can yield a singular matrix
    # whose `eigvals` are NaN, tripping `check_inf_nan`).
    dense_init(rng, T, d...) = rand_sparse(rng, T, d...; sparsity = 0.5)
    for α in (1.0, 0.3)
        rng = MersenneTwister(13)
        in_dim, res_dim, T_steps = 2, 16, 12

        ce = ContinuousESN(
            in_dim, res_dim, (0.0, Float64(T_steps)), Euler();
            leak_coefficient = α, use_bias = true, T = Float64, dt = 1.0,
            init_reservoir = dense_init
        )
        rc = ReservoirComputer(ce, LinearReadout(res_dim => 1))
        ps, st = setup(MersenneTwister(0), rc)

        data = randn(rng, in_dim, T_steps)
        cont_states, _ = collectstates(rc, data, ps, st)

        W_r = ps.reservoir.W_r
        W_in = ps.reservoir.W_in
        b = ps.reservoir.b
        disc_states = zeros(Float64, res_dim, T_steps)
        x = zeros(Float64, res_dim)
        for (k, u_col) in enumerate(eachcol(data))
            x = (1 - α) .* x .+ α .* tanh.(W_r * x + W_in * u_col + b)
            disc_states[:, k] = x
        end
        @test cont_states ≈ disc_states atol = 1.0e-10
    end
end

# ---------------------------------------------------------------------------
# 6. Teacher-forced predict + autoregressive predict shape/determinism
# ---------------------------------------------------------------------------

@testset "ContinuousESN: teacher-forced predict" begin
    rng = MersenneTwister(21)
    in_dim, res_dim, out_dim, T_steps = 2, 12, 3, 10
    ce = ContinuousESN(
        in_dim, res_dim, (0.0, 2.0), Tsit5();
        reltol = 1.0e-8, abstol = 1.0e-10
    )
    rc = ReservoirComputer(ce, LinearReadout(res_dim => out_dim))
    ps, st = setup(rng, rc)

    data = randn(Float32, in_dim, T_steps)
    tf1, _ = predict(rc, data, ps, st)
    tf2, _ = predict(rc, data, ps, st)
    @test size(tf1) == (out_dim, T_steps)
    @test all(isfinite, tf1)
    @test tf1 ≈ tf2
end

@testset "ContinuousESN: autoregressive predict" begin
    rng = MersenneTwister(22)
    # Autoregressive rollout feeds outputs back as inputs, so in_dim == out_dim.
    dim, res_dim, steps = 3, 12, 5
    ce = ContinuousESN(
        dim, res_dim, (0.0, 1.0), Tsit5();
        reltol = 1.0e-8, abstol = 1.0e-10
    )
    rc = ReservoirComputer(ce, LinearReadout(res_dim => dim))
    ps, st = setup(rng, rc)

    init = randn(Float32, dim)
    ar1, _ = predict(rc, steps, ps, st; initialdata = init)
    ar2, _ = predict(rc, steps, ps, st; initialdata = init)
    @test size(ar1) == (dim, steps)
    @test all(isfinite, ar1)
    @test ar1 ≈ ar2
end

# ---------------------------------------------------------------------------
# 7. Compatibility with state modifiers
# ---------------------------------------------------------------------------

@testset "ContinuousESN: state modifiers compose" begin
    rng = MersenneTwister(31)
    in_dim, res_dim, T_steps = 2, 10, 8
    ce = ContinuousESN(
        in_dim, res_dim, (0.0, 1.0), Tsit5();
        reltol = 1.0e-8, abstol = 1.0e-10
    )
    rc_plain = ReservoirComputer(ce, LinearReadout(res_dim => 1))
    rc_mod = ReservoirComputer(ce, (NLAT2(),), LinearReadout(res_dim => 1))

    ps_p, st_p = setup(MersenneTwister(0), rc_plain)
    ps_m, st_m = setup(MersenneTwister(0), rc_mod)
    data = randn(Float32, in_dim, T_steps)

    sp, _ = collectstates(rc_plain, data, ps_p, st_p)
    sm, _ = collectstates(rc_mod, data, ps_m, st_m)
    @test size(sm) == size(sp)
    @test all(isfinite, sm)
    @test sm != sp
end

# ---------------------------------------------------------------------------
# 8. Eltype propagates through `T` kwarg
# ---------------------------------------------------------------------------

@testset "ContinuousESN: T kwarg controls matrix eltype" begin
    rng = MersenneTwister(43)
    ce64 = ContinuousESN(2, 32, (0.0, 1.0), Tsit5(); T = Float64)
    rc = ReservoirComputer(ce64, LinearReadout(32 => 1))
    ps, _ = setup(rng, rc)
    @test eltype(ps.reservoir.W_r) == Float64
    @test eltype(ps.reservoir.W_in) == Float64
    @test typeof(ps.reservoir.leak_coefficient) == Float64
end
