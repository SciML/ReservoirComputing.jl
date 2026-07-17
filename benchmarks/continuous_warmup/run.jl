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
    only = nothing
    for a in args
        if startswith(a, "--only=")
            only = split(a[8:end], ','; keepempty = false)
        end
    end
    return (; smoke, only)
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
            "max_abs_diff", "match_ok", "u0_norm", "has_carry")
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
    cfg = (
        smoke = opts.smoke,
        seed = 17,
        n_res = opts.smoke ? 80 : 300,
        ridge = 1.0e-6,
        data = data,
    )

    println("Continuous warmup investigation")
    println("  smoke=$(cfg.smoke) n_res=$(cfg.n_res) train=$(data.train_len) ",
        "predict=$(data.predict_len)")
    println("  only=$(opts.only === nothing ? "all" : join(opts.only, ","))")

    rows = run_experiments(cfg; only = opts.only)

    json_path = joinpath(results_dir, "summary.json")
    open(json_path, "w") do io
        JSON.print(io, rows, 2)
    end

    md_path = joinpath(results_dir, "summary.md")
    write(md_path, rows_to_markdown(rows))

    println("\nWrote $json_path")
    println("Wrote $md_path")
    println("Edit results/FINDINGS.md with interpretation after review.")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
