using Documenter, ReservoirComputing

#cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
#cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"
include("pages.jl")

makedocs(modules = [ReservoirComputing],
    sitename = "ReservoirComputing.jl",
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:missing_docs],
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/ReservoirComputing/stable/"),
    pages = pages)

deploydocs(repo = "github.com/SciML/ReservoirComputing.jl.git";
    push_preview = true)
