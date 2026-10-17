#!/usr/bin/env julia
# Continuous AR predict warmup investigation entrypoint.
#
# Usage:
#   julia --project=. run.jl --smoke
#   julia --project=. run.jl
#   julia --project=. run.jl --only=E1,E6

using Pkg
Pkg.instantiate()

using Random
using JSON
using Dates
using SciMLBase
using DataInterpolations
using OrdinaryDiffEqTsit5
using OrdinaryDiffEq
using ReservoirComputing

const ROOT = @__DIR__
include(joinpath(ROOT, "src", "metrics.jl"))
include(joinpath(ROOT, "src", "data.jl"))
include(joinpath(ROOT, "src", "models.jl"))
include(joinpath(ROOT, "src", "predict_variants.jl"))
include(joinpath(ROOT, "src", "experiments.jl"))

function parse_args(args)
    smoke = "--smoke" in args
    # Alias kept for clarity in docs/CI; default without --smoke is full.
    full = "--full" in args || !smoke
    only = nothing
    n_res = nothing
    for a in args
        if startswith(a, "--only=")
            only = split(a[8:end], ','; keepempty = false)
        elseif startswith(a, "--n-res=")
            n_res = parse(Int, a[9:end])
        end
    end
    return (; smoke, full, only, n_res)
end

function rows_to_markdown(rows)
    io = IOBuffer()
    println(io, "# Continuous warmup investigation results")
    println(io)
    println(io, "Generated: $(Dates.now())")
    println(io)
    println(io, "| exp | variant | model | nrmse | nrmse_global | vpt_lyap | notes |")
    println(io, "|-----|---------|-------|-------|--------------|----------|-------|")
    for r in rows
        nrmse = get(r, "nrmse", "")
        nrmse_g = get(r, "nrmse_global", "")
        vpt = get(r, "vpt_lyap", "")
        notes = String[]
        for k in ("warmup_len", "horizon_steps", "horizon_lyap", "washout",
            "max_abs_diff", "match_ok", "u0_norm", "has_carry",
            "wall_train_s", "wall_ar_s", "wall_warmup_collect_s", "wall_s", "mode")
            haskey(r, k) && push!(notes, "$k=$(r[k])")
        end
        println(
            io,
            "| $(r["experiment"]) | $(r["variant"]) | $(r["model"]) | ",
            "$(nrmse) | $(nrmse_g) | $(vpt) | $(join(notes, "; ")) |",
        )
    end
    return String(take!(io))
end

function main(args)
    opts = parse_args(args)
    results_dir = joinpath(ROOT, "results")
    mkpath(results_dir)

    data = opts.smoke ? make_lorenz_smoke() : make_lorenz()
    n_res = something(opts.n_res, opts.smoke ? 80 : 300)
    mode = opts.smoke ? "smoke" : "full"
    cfg = (
        smoke = opts.smoke,
        mode = mode,
        seed = 17,
        n_res = n_res,
        ridge = 1.0e-6,
        data = data,
    )

    println("Continuous warmup investigation")
    println("  mode=$(mode) n_res=$(cfg.n_res) train=$(data.train_len) ",
        "predict=$(data.predict_len)  (t_λ span ≈ ",
        round(data.predict_len * data.dt * data.λ_max; digits = 1), ")")
    println("  only=$(opts.only === nothing ? "all" : join(opts.only, ","))")
    println("  HPs: radius=0.9, input_scale=0.1, bias_scale=0.05, ridge=$(cfg.ridge)")

    t_all0 = time_ns()
    rows = run_experiments(cfg; only = opts.only)
    wall_all = (time_ns() - t_all0) / 1.0e9
    for r in rows
        r["mode"] = mode
        r["n_res_cfg"] = n_res
    end
    push!(
        rows,
        Dict{String, Any}(
            "experiment" => "META",
            "variant" => "suite_wall_s",
            "model" => "harness",
            "mode" => mode,
            "wall_s" => wall_all,
            "n_res" => n_res,
            "train_len" => data.train_len,
            "predict_len" => data.predict_len,
        ),
    )

    tag = mode
    json_path = joinpath(results_dir, "summary_$(tag).json")
    open(json_path, "w") do io
        JSON.print(io, rows, 2)
    end
    # Keep untagged names as the latest run for convenience.
    cp(json_path, joinpath(results_dir, "summary.json"); force = true)

    md_path = joinpath(results_dir, "summary_$(tag).md")
    write(md_path, rows_to_markdown(rows))
    cp(md_path, joinpath(results_dir, "summary.md"); force = true)

    println("\nSuite wall time: $(round(wall_all; digits=1)) s")
    println("Wrote $json_path")
    println("Wrote $md_path")
    println("Update results/FINDINGS.md with interpretation after review.")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
