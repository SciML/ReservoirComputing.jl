using Dates
using JSON
using Random
using Statistics
using LinearAlgebra

"""Accumulate one experiment row."""
function _row(; experiment, variant, model, kwargs...)
    d = Dict{String, Any}(
        "experiment" => string(experiment),
        "variant" => string(variant),
        "model" => string(model),
        "timestamp" => string(Dates.now()),
    )
    for (k, v) in pairs(kwargs)
        d[string(k)] = v
    end
    return d
end

function _score(pred, truth, data)
    return (
        nrmse = nrmse(pred, truth),
        nrmse_global = nrmse_global(pred, truth),
        vpt = valid_prediction_time(
            pred, truth; dt = data.dt, λ_max = data.λ_max, threshold = 0.5
        ),
    )
end

function _horizons(predict_len)
    # steps ≈ Lyapunov times * λ_max / dt  →  t_λ = steps * dt * λ_max
    # invert: steps = t_λ / (dt * λ_max)
    return nothing  # filled per-call with data
end

function lyap_horizons(data; t_λs = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0))
    steps = Int[]
    for tλ in t_λs
        h = max(1, round(Int, tλ / (data.dt * data.λ_max)))
        h ≤ data.predict_len && push!(steps, h)
    end
    return unique(steps)
end

# ---------------------------------------------------------------------------
# E1 — ContinuousESN cold vs train-terminal warm
# ---------------------------------------------------------------------------

function run_E1(cfg)
    println("\n=== E1: ContinuousESN cold vs train-terminal warm ===")
    rng = MersenneTwister(cfg.seed)
    data = cfg.data
    n_res = cfg.n_res

    m_train = build_continuous_esn(n_res, data.train_len)
    m_pred = build_continuous_esn(n_res, data.predict_len)
    tr = train_pair(m_train, m_pred, data; ridge = cfg.ridge, rng = rng)
    ps_ar, st_ar = align_pred_params(
        tr.ps_train, tr.st_train, tr.ps_pred, tr.st_pred
    )

    cold, _ = predict_ar_cold(
        tr.model_pred, data.predict_len, ps_ar, st_ar;
        initialdata = data.test_data[:, 1],
    )
    sc_cold = _score(cold, data.test_data, data)

    u0 = raw_terminal_ode_state(
        tr.model_train, data.input_data, tr.ps_train, tr.st_train
    )
    warm, _ = predict_ar_seeded(
        tr.model_pred, data.predict_len, ps_ar, st_ar;
        initialdata = data.test_data[:, 1],
        initial_state = u0,
    )
    sc_warm = _score(warm, data.test_data, data)

    # Sanity: seeded with zeros should match package cold path
    zero_u0 = zeros(eltype(u0), length(u0))
    seeded_zero, _ = predict_ar_seeded(
        tr.model_pred, data.predict_len, ps_ar, st_ar;
        initialdata = data.test_data[:, 1],
        initial_state = zero_u0,
    )
    match_cold = maximum(abs.(seeded_zero .- cold))

    rows = [
        _row(;
            experiment = "E1",
            variant = "cold_package_predict",
            model = "ContinuousESN",
            n_res = n_res,
            nrmse = sc_cold.nrmse,
            nrmse_global = sc_cold.nrmse_global,
            vpt_lyap = sc_cold.vpt,
            train_len = data.train_len,
            predict_len = data.predict_len,
        ),
        _row(;
            experiment = "E1",
            variant = "warm_train_terminal_u0",
            model = "ContinuousESN",
            n_res = n_res,
            nrmse = sc_warm.nrmse,
            nrmse_global = sc_warm.nrmse_global,
            vpt_lyap = sc_warm.vpt,
            train_len = data.train_len,
            predict_len = data.predict_len,
            u0_norm = norm(u0),
        ),
        _row(;
            experiment = "E1",
            variant = "seeded_zero_matches_cold_maxabs",
            model = "ContinuousESN",
            n_res = n_res,
            max_abs_diff = match_cold,
            match_ok = match_cold < 1.0e-8,
        ),
    ]

    @info "E1 cold nrmse=$(round(sc_cold.nrmse; digits=4)) warm=$(round(sc_warm.nrmse; digits=4)) " *
        "Δ=$(round(sc_cold.nrmse - sc_warm.nrmse; digits=4)) zero_match=$(match_cold)"
    return rows
end

# ---------------------------------------------------------------------------
# E2 — SciMLProblemReservoir eq.5 cold vs warm
# ---------------------------------------------------------------------------

function run_E2(cfg)
    println("\n=== E2: SciMLProblemReservoir (eq.5) cold vs warm ===")
    rng = MersenneTwister(cfg.seed)
    data = cfg.data
    n_res = cfg.n_res

    rng2 = MersenneTwister(cfg.seed)
    m_train = build_sciml_eq5(rng2, n_res, data.train_len)
    # Same Wr/Win/b as train; only tspan / u0 sized for predict length.
    res_tr = m_train.reservoir
    tspan_p = (0.0, Float64(data.predict_len))
    prob_p = remake(res_tr.prob; u0 = zeros(n_res), tspan = tspan_p)
    res_p = SciMLProblemReservoir(
        prob_p, TerminalStateSampling(), tspan_p, Tsit5();
        reltol = RELTOL, abstol = ABSTOL,
    )
    m_pred = ReservoirComputer(res_p, m_train.states_modifiers, LinearReadout(n_res => 3))

    tr = train_pair(m_train, m_pred, data; ridge = cfg.ridge, rng = MersenneTwister(cfg.seed))

    cold, _ = predict_ar_cold(
        tr.model_pred, data.predict_len, tr.ps_pred, tr.st_pred;
        initialdata = data.test_data[:, 1],
    )
    sc_cold = _score(cold, data.test_data, data)

    u0 = raw_terminal_ode_state(
        tr.model_train, data.input_data, tr.ps_train, tr.st_train
    )
    warm, _ = predict_ar_seeded(
        tr.model_pred, data.predict_len, tr.ps_pred, tr.st_pred;
        initialdata = data.test_data[:, 1],
        initial_state = u0,
    )
    sc_warm = _score(warm, data.test_data, data)

    # remake public-style: set pred.prob.u0 then package predict
    res_pred = tr.model_pred.reservoir
    res_warm = SciMLProblemReservoir(
        remake(res_pred.prob; u0 = u0),
        res_pred.sampler,
        res_pred.tspan,
        res_pred.args,
        res_pred.kwargs,
    )
    m_remake = ReservoirComputer(
        res_warm, tr.model_pred.states_modifiers, tr.model_pred.readout
    )
    remake_out, _ = predict(
        m_remake, data.predict_len, tr.ps_pred, tr.st_pred;
        initialdata = data.test_data[:, 1],
    )
    sc_remake = _score(remake_out, data.test_data, data)
    match_warm = maximum(abs.(remake_out .- warm))

    rows = [
        _row(;
            experiment = "E2",
            variant = "cold_package_predict",
            model = "SciMLProblemReservoir",
            n_res = n_res,
            nrmse = sc_cold.nrmse,
            nrmse_global = sc_cold.nrmse_global,
            vpt_lyap = sc_cold.vpt,
        ),
        _row(;
            experiment = "E2",
            variant = "warm_train_terminal_u0",
            model = "SciMLProblemReservoir",
            n_res = n_res,
            nrmse = sc_warm.nrmse,
            nrmse_global = sc_warm.nrmse_global,
            vpt_lyap = sc_warm.vpt,
            u0_norm = norm(u0),
        ),
        _row(;
            experiment = "E2",
            variant = "remake_prob_u0_package_predict",
            model = "SciMLProblemReservoir",
            n_res = n_res,
            nrmse = sc_remake.nrmse,
            nrmse_global = sc_remake.nrmse_global,
            vpt_lyap = sc_remake.vpt,
            max_abs_diff_vs_seeded = match_warm,
        ),
    ]

    @info "E2 cold=$(round(sc_cold.nrmse; digits=4)) warm=$(round(sc_warm.nrmse; digits=4)) " *
        "remake=$(round(sc_remake.nrmse; digits=4))"
    return rows
end

# ---------------------------------------------------------------------------
# E3 — warmup length sweep
# ---------------------------------------------------------------------------

function run_E3(cfg)
    println("\n=== E3: warmup length sweep (ContinuousESN) ===")
    rng = MersenneTwister(cfg.seed)
    data = cfg.data
    n_res = cfg.n_res

    m_train = build_continuous_esn(n_res, data.train_len)
    m_pred = build_continuous_esn(n_res, data.predict_len)
    tr = train_pair(m_train, m_pred, data; ridge = cfg.ridge, rng = rng)

    ps_ar, st_ar = align_pred_params(
        tr.ps_train, tr.st_train, tr.ps_pred, tr.st_pred
    )

    Ks = cfg.smoke ? [0, 10, 50, 100, 200] : [0, 10, 50, 100, 250, 500, 1000, 2000]
    Ks = filter(k -> k == 0 || k ≤ data.train_len, Ks)

    rows = Dict{String, Any}[]
    for K in Ks
        if K == 0
            pred, _ = predict_ar_cold(
                tr.model_pred, data.predict_len, ps_ar, st_ar;
                initialdata = data.test_data[:, 1],
            )
            variant = "K=0_cold"
        else
            warm_data = data.input_data[:, (end - K + 1):end]
            # collectstates on pred model needs tspan matching warm_data length
            # ContinuousESN tspan is fixed to predict_len — length mismatch!
            # Use train model for warmup collect (tspan = train_len), then AR on pred.
            u0 = raw_terminal_ode_state(
                tr.model_train, warm_data, tr.ps_train, tr.st_train
            )
            pred, _ = predict_ar_seeded(
                tr.model_pred, data.predict_len, ps_ar, st_ar;
                initialdata = data.test_data[:, 1],
                initial_state = u0,
            )
            variant = "K=$(K)"
        end
        sc = _score(pred, data.test_data, data)
        push!(
            rows,
            _row(;
                experiment = "E3",
                variant = variant,
                model = "ContinuousESN",
                warmup_len = K,
                nrmse = sc.nrmse,
                nrmse_global = sc.nrmse_global,
                vpt_lyap = sc.vpt,
            )
        )
        @info "E3 K=$K nrmse=$(round(sc.nrmse; digits=4)) vpt=$(round(sc.vpt; digits=3))"
    end
    return rows
end

# ---------------------------------------------------------------------------
# E4 — seed variants
# ---------------------------------------------------------------------------

function run_E4(cfg)
    println("\n=== E4: seed variants (ContinuousESN) ===")
    rng = MersenneTwister(cfg.seed)
    data = cfg.data
    n_res = cfg.n_res

    m_train = build_continuous_esn(n_res, data.train_len)
    m_pred = build_continuous_esn(n_res, data.predict_len)
    tr = train_pair(m_train, m_pred, data; ridge = cfg.ridge, rng = rng)
    ps_ar, st_ar = align_pred_params(
        tr.ps_train, tr.st_train, tr.ps_pred, tr.st_pred
    )

    u_train = raw_terminal_ode_state(
        tr.model_train, data.input_data, tr.ps_train, tr.st_train
    )
    # test-prefix warm: first min(100, predict_len÷2) of test via teacher force
    # but test is the target for AR — use teacher force on true test prefix
    # as "oracle warm" upper bound
    K_prefix = min(100, max(10, data.predict_len ÷ 5))
    # For raw terminal on train model, use train-length tspan with a short
    # series of test inputs placed as if they were a mini drive.
    # Use train model with test prefix as data (grid scales to cell.tspan).
    u_oracle = raw_terminal_ode_state(
        tr.model_train,
        data.test_data[:, 1:K_prefix],
        tr.ps_train,
        tr.st_train,
    )

    seeds = Dict(
        "zeros" => zeros(n_res),
        "randn" => randn(MersenneTwister(0), n_res),
        "train_terminal" => u_train,
        "oracle_test_prefix" => u_oracle,
        "shuffled_train_terminal" => u_train[randperm(MersenneTwister(1), n_res)],
    )

    rows = Dict{String, Any}[]
    for (name, u0) in seeds
        pred, _ = predict_ar_seeded(
            tr.model_pred, data.predict_len, ps_ar, st_ar;
            initialdata = data.test_data[:, 1],
            initial_state = Vector{Float64}(u0),
        )
        sc = _score(pred, data.test_data, data)
        push!(
            rows,
            _row(;
                experiment = "E4",
                variant = name,
                model = "ContinuousESN",
                nrmse = sc.nrmse,
                nrmse_global = sc.nrmse_global,
                vpt_lyap = sc.vpt,
                u0_norm = norm(u0),
            )
        )
        @info "E4 $name nrmse=$(round(sc.nrmse; digits=4))"
    end
    return rows
end

# ---------------------------------------------------------------------------
# E5 — horizon curves cold vs warm
# ---------------------------------------------------------------------------

function run_E5(cfg)
    println("\n=== E5: horizon NRMSE curves ===")
    rng = MersenneTwister(cfg.seed)
    data = cfg.data
    n_res = cfg.n_res

    m_train = build_continuous_esn(n_res, data.train_len)
    m_pred = build_continuous_esn(n_res, data.predict_len)
    tr = train_pair(m_train, m_pred, data; ridge = cfg.ridge, rng = rng)
    ps_ar, st_ar = align_pred_params(
        tr.ps_train, tr.st_train, tr.ps_pred, tr.st_pred
    )

    cold, _ = predict_ar_cold(
        tr.model_pred, data.predict_len, ps_ar, st_ar;
        initialdata = data.test_data[:, 1],
    )
    u0 = raw_terminal_ode_state(
        tr.model_train, data.input_data, tr.ps_train, tr.st_train
    )
    warm, _ = predict_ar_seeded(
        tr.model_pred, data.predict_len, ps_ar, st_ar;
        initialdata = data.test_data[:, 1],
        initial_state = u0,
    )

    hs = lyap_horizons(data)
    rows = Dict{String, Any}[]
    for h in hs
        tλ = h * data.dt * data.λ_max
        for (variant, pred) in (("cold", cold), ("warm_train_terminal", warm))
            sc = nrmse(@view(pred[:, 1:h]), @view(data.test_data[:, 1:h]))
            push!(
                rows,
                _row(;
                    experiment = "E5",
                    variant = variant,
                    model = "ContinuousESN",
                    horizon_steps = h,
                    horizon_lyap = tλ,
                    nrmse = sc,
                )
            )
        end
        @info "E5 h=$h (tλ=$(round(tλ; digits=2))) cold=$(round(nrmse(cold[:,1:h], data.test_data[:,1:h]); digits=4)) " *
            "warm=$(round(nrmse(warm[:,1:h], data.test_data[:,1:h]); digits=4))"
    end
    return rows
end

# ---------------------------------------------------------------------------
# E6 — does st after train hold continuous carry?
# ---------------------------------------------------------------------------

function run_E6(cfg)
    println("\n=== E6: inspect st after train! / collectstates ===")
    rng = MersenneTwister(cfg.seed)
    data = cfg.data
    n_res = cfg.n_res

    m = build_continuous_esn(n_res, data.train_len)
    ps, st0 = setup(rng, m)
    ps, st_train = train!(
        m, data.input_data, data.target_data, ps, st0, StandardRidge(cfg.ridge)
    )
    _, st_collect = collectstates(m, data.input_data, ps, st_train)

    d0 = inspect_st_reservoir(st0)
    d1 = inspect_st_reservoir(st_train)
    d2 = inspect_st_reservoir(st_collect)

    m_disc = build_discrete_esn(n_res)
    ps_d, st_d0 = setup(MersenneTwister(cfg.seed), m_disc)
    ps_d, st_d = train!(
        m_disc, data.input_data, data.target_data, ps_d, st_d0, StandardRidge(cfg.ridge)
    )
    d_disc = inspect_st_reservoir(st_d)

    rows = [
        _row(;
            experiment = "E6",
            variant = "continuous_st0",
            model = "ContinuousESN",
            st_type = d0.type,
            keys = something(d0.keys, []),
            has_carry = d0.has_carry,
        ),
        _row(;
            experiment = "E6",
            variant = "continuous_st_after_train",
            model = "ContinuousESN",
            st_type = d1.type,
            keys = something(d1.keys, []),
            has_carry = d1.has_carry,
            summary = d1.summary,
        ),
        _row(;
            experiment = "E6",
            variant = "continuous_st_after_collectstates",
            model = "ContinuousESN",
            st_type = d2.type,
            keys = something(d2.keys, []),
            has_carry = d2.has_carry,
            summary = d2.summary,
        ),
        _row(;
            experiment = "E6",
            variant = "discrete_st_after_train",
            model = "ESN",
            st_type = d_disc.type,
            keys = something(d_disc.keys, []),
            has_carry = d_disc.has_carry,
            summary = d_disc.summary,
        ),
    ]

    @info "E6 continuous has_carry after train=$(d1.has_carry); discrete has_carry=$(d_disc.has_carry)"
    return rows
end

# ---------------------------------------------------------------------------
# E7 — discrete ESN control
# ---------------------------------------------------------------------------

function run_E7(cfg)
    println("\n=== E7: discrete ESN cold vs warm st ===")
    rng = MersenneTwister(cfg.seed)
    data = cfg.data
    n_res = cfg.n_res

    m = build_discrete_esn(n_res)
    ps, st0 = setup(rng, m)
    ps, st_train = train!(
        m, data.input_data, data.target_data, ps, st0, StandardRidge(cfg.ridge)
    )

    # cold: fresh states, only readout trained
    _, st_cold = setup(MersenneTwister(cfg.seed), m)
    st_cold = merge(st_cold, (readout = st_train.readout,))
    cold, _ = predict(
        m, data.predict_len, ps, st_cold; initialdata = data.test_data[:, 1]
    )

    # warm: continue from post-train states
    warm, _ = predict(
        m, data.predict_len, ps, st_train; initialdata = data.test_data[:, 1]
    )

    # re-drive last train window then AR (like warmup_data)
    _, st_rewarm = collectstates(m, data.input_data, ps, st0)
    st_rewarm = merge(st_rewarm, (readout = st_train.readout,))
    rewarm, _ = predict(
        m, data.predict_len, ps, st_rewarm; initialdata = data.test_data[:, 1]
    )

    sc_c = _score(cold, data.test_data, data)
    sc_w = _score(warm, data.test_data, data)
    sc_r = _score(rewarm, data.test_data, data)

    rows = [
        _row(;
            experiment = "E7",
            variant = "cold_fresh_st",
            model = "ESN",
            nrmse = sc_c.nrmse,
            vpt_lyap = sc_c.vpt,
        ),
        _row(;
            experiment = "E7",
            variant = "warm_st_after_train",
            model = "ESN",
            nrmse = sc_w.nrmse,
            vpt_lyap = sc_w.vpt,
        ),
        _row(;
            experiment = "E7",
            variant = "rewarm_collectstates_then_ar",
            model = "ESN",
            nrmse = sc_r.nrmse,
            vpt_lyap = sc_r.vpt,
        ),
    ]
    @info "E7 cold=$(round(sc_c.nrmse; digits=4)) warm_train_st=$(round(sc_w.nrmse; digits=4)) " *
        "rewarm=$(round(sc_r.nrmse; digits=4))"
    return rows
end

# ---------------------------------------------------------------------------
# E8 — washout interaction
# ---------------------------------------------------------------------------

function run_E8(cfg)
    println("\n=== E8: washout + warm (ContinuousESN) ===")
    rng = MersenneTwister(cfg.seed)
    data = cfg.data
    n_res = cfg.n_res
    washout = cfg.smoke ? 50 : 200

    m_train = build_continuous_esn(n_res, data.train_len)
    m_pred = build_continuous_esn(n_res, data.predict_len)
    tr = train_pair(
        m_train, m_pred, data; ridge = cfg.ridge, rng = rng, washout = washout
    )
    ps_ar, st_ar = align_pred_params(
        tr.ps_train, tr.st_train, tr.ps_pred, tr.st_pred
    )

    cold, _ = predict_ar_cold(
        tr.model_pred, data.predict_len, ps_ar, st_ar;
        initialdata = data.test_data[:, 1],
    )
    u_full = raw_terminal_ode_state(
        tr.model_train, data.input_data, tr.ps_train, tr.st_train
    )
    # post-washout tail only
    tail = data.input_data[:, (washout + 1):end]
    u_tail = if size(tail, 2) ≥ 2
        raw_terminal_ode_state(tr.model_train, tail, tr.ps_train, tr.st_train)
    else
        u_full
    end

    warm_full, _ = predict_ar_seeded(
        tr.model_pred, data.predict_len, ps_ar, st_ar;
        initialdata = data.test_data[:, 1], initial_state = u_full,
    )
    warm_tail, _ = predict_ar_seeded(
        tr.model_pred, data.predict_len, ps_ar, st_ar;
        initialdata = data.test_data[:, 1], initial_state = u_tail,
    )

    rows = [
        _row(;
            experiment = "E8",
            variant = "cold",
            model = "ContinuousESN",
            washout = washout,
            nrmse = nrmse(cold, data.test_data),
        ),
        _row(;
            experiment = "E8",
            variant = "warm_full_train",
            model = "ContinuousESN",
            washout = washout,
            nrmse = nrmse(warm_full, data.test_data),
        ),
        _row(;
            experiment = "E8",
            variant = "warm_post_washout_tail",
            model = "ContinuousESN",
            washout = washout,
            nrmse = nrmse(warm_tail, data.test_data),
        ),
    ]
    @info "E8 washout=$washout cold=$(round(rows[1]["nrmse"]; digits=4)) " *
        "full=$(round(rows[2]["nrmse"]; digits=4)) tail=$(round(rows[3]["nrmse"]; digits=4))"
    return rows
end

const EXPERIMENT_FUNS = Dict(
    "E1" => run_E1,
    "E2" => run_E2,
    "E3" => run_E3,
    "E4" => run_E4,
    "E5" => run_E5,
    "E6" => run_E6,
    "E7" => run_E7,
    "E8" => run_E8,
)

function run_experiments(cfg; only = nothing)
    ids = only === nothing ? collect(keys(EXPERIMENT_FUNS)) : only
    sort!(ids)
    all_rows = Dict{String, Any}[]
    for id in ids
        haskey(EXPERIMENT_FUNS, id) || throw(ArgumentError("unknown experiment $id"))
        append!(all_rows, EXPERIMENT_FUNS[id](cfg))
    end
    return all_rows
end
