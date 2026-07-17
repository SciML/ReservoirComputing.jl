using Random
using LinearAlgebra
using ReservoirComputing
using SciMLBase
using OrdinaryDiffEqTsit5

const RELTOL = 1.0e-6
const ABSTOL = 1.0e-8

init_input_f64(rng, dims...) = scaled_rand(rng, Float64, dims...)
init_reservoir_f64(rng, dims...) = rand_sparse(rng, Float64, dims...)
init_bias_f64(rng, dims...) = zeros(Float64, dims...)
init_state_f64(rng, dims...) = zeros(Float64, dims...)

function f64_inits(; use_bias = true)
    return (
        use_bias = use_bias,
        init_input = init_input_f64,
        init_reservoir = init_reservoir_f64,
        init_bias = init_bias_f64,
        init_state = init_state_f64,
        reltol = RELTOL,
        abstol = ABSTOL,
    )
end

"""
    build_continuous_esn(n_res, n_steps; radius=0.9, kwargs...)

`tspan = (0, n_steps)` so one sample ≈ one unit of reservoir time
(tutorial / #456 convention).

`radius` is applied to `rand_sparse` (spectral radius of `W_r`). Continuous
eq. (5) needs a tighter bound than many discrete demos; 0.9 matches the
#456 Lorenz probe.
"""
function build_continuous_esn(
        n_res::Integer,
        n_steps::Integer;
        state_modifiers = (NLAT2(),),
        use_bias::Bool = true,
        solver = Tsit5(),
        radius::Float64 = 0.9,
        extra...,
    )
    init_res = (rng, dims...) -> rand_sparse(rng, Float64, dims...; radius = radius)
    return ContinuousESN(
        3, n_res, 3, (0.0, Float64(n_steps)), solver;
        f64_inits(; use_bias)...,
        init_reservoir = init_res,
        state_modifiers = state_modifiers,
        extra...,
    )
end

"""
Copy trained reservoir + readout parameters into a predict-length model
setup so cold/warm AR use the **same** `W_in` / `W_r` / bias / `W_out`.
"""
function align_pred_params(ps_train, st_train, ps_pred, st_pred)
    ps = merge(ps_pred, (reservoir = ps_train.reservoir, readout = ps_train.readout))
    st = merge(st_pred, (readout = st_train.readout,))
    return ps, st
end

"""
    eq5_rhs!(dx, x, p, t)

Lukoševičius §3.2.6 eq. (5) for hand-rolled `SciMLProblemReservoir`.
"""
function eq5_rhs!(dx, x, p, t)
    input_t = p.input(t)
    # Match hand-rolled #456 / test style: Wr, Win, b in `prob.p`.
    dx .= .-x .+ tanh.(p.Wr * x .+ p.Win * input_t .+ p.b)
    return nothing
end

"""
    build_sciml_eq5(rng, n_res, n_steps; radius=0.9, …)

Sparse random Wr rescaled to spectral radius `radius` (continuous ESP-ish).
"""
function build_sciml_eq5(
        rng::AbstractRNG,
        n_res::Integer,
        n_steps::Integer;
        radius::Float64 = 0.9,
        input_scale::Float64 = 0.1,
        use_bias::Bool = true,
        sparsity::Float64 = 6 / n_res,
        state_modifiers = (NLAT2(),),
        solver = Tsit5(),
    )
    Wr_raw = randn(rng, n_res, n_res)
    mask = rand(rng, n_res, n_res) .< sparsity
    Wr_sparse = Wr_raw .* mask
    ρ = maximum(abs.(eigvals(Matrix(Wr_sparse))))
    ρ = ρ > 0 ? ρ : 1.0
    Wr = (radius / ρ) .* Wr_sparse
    Win = input_scale .* randn(rng, n_res, 3)
    bias = use_bias ? (0.05 .* randn(rng, n_res)) : zeros(n_res)
    p0 = (Wr = Wr, Win = Win, b = bias)

    tspan = (0.0, Float64(n_steps))
    u0 = zeros(n_res)
    prob = ODEProblem(eq5_rhs!, u0, tspan, p0)
    res = SciMLProblemReservoir(
        prob, TerminalStateSampling(), tspan, solver;
        reltol = RELTOL, abstol = ABSTOL,
    )
    return ReservoirComputer(res, state_modifiers, LinearReadout(n_res => 3))
end

function build_discrete_esn(
        n_res::Integer;
        state_modifiers = (NLAT2(),),
        use_bias::Bool = true,
        radius::Float64 = 0.9,
    )
    return ESN(
        3, n_res, 3;
        use_bias = use_bias,
        init_input = init_input_f64,
        init_reservoir = (rng, dims...) -> rand_sparse(
            rng, Float64, dims...; radius = radius
        ),
        init_bias = init_bias_f64,
        init_state = init_state_f64,
        state_modifiers = state_modifiers,
    )
end

"""
    train_pair(model_train, model_pred, data; washout, ridge, rng)

Train on `model_train`, copy readout into a fresh `model_pred` parameter
set (same pattern as continuous tutorial).
"""
function train_pair(
        model_train,
        model_pred,
        data;
        washout::Int = 0,
        ridge::Float64 = 1.0e-6,
        rng = MersenneTwister(17),
    )
    ps, st = setup(rng, model_train)
    ps, st = train!(
        model_train, data.input_data, data.target_data, ps, st,
        StandardRidge(ridge); washout = washout
    )

    ps_pred, st_pred = setup(rng, model_pred)
    ps_pred = merge(ps_pred, (readout = ps.readout,))
    st_pred = merge(st_pred, (readout = st.readout,))

    return (
        ps_train = ps,
        st_train = st,
        ps_pred = ps_pred,
        st_pred = st_pred,
        model_train = model_train,
        model_pred = model_pred,
    )
end
