using Documenter, DocumenterCitations, ReservoirCellularAutomata

#cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml"; force=true)
#cp("./docs/Project.toml", "./docs/src/assets/Project.toml"; force=true)

ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"
include("pages.jl")
mathengine = Documenter.MathJax()

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:authoryear
)

makedocs(; modules=[ReservoirCellularAutomata],
    sitename="ReservoirCellularAutomata.jl",
    authors="Francesco Martinuzzi",
    clean=true, doctest=false, linkcheck=true,
    plugins=[bib],
    format=Documenter.HTML(;
        mathengine,
        assets=["assets/favicon.ico"],
        canonical="https://docs.sciml.ai/ReservoirComputing/ReservoirCellularAutomata/"),
    pages=pages
)

deploydocs(
    repo="github.com/SciML/ReservoirComputing.jl.git",
    target="build",
    branch="docs-reca",
    devbranch="master",
    tag_prefix="ReservoirCellularAutomata-",
    push_preview=true,
)
