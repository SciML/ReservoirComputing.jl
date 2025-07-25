using ReservoirCellularAutomata
using Documenter

DocMeta.setdocmeta!(ReservoirCellularAutomata, :DocTestSetup,
    :(using ReservoirCellularAutomata); recursive = true)

makedocs(;
    modules = [ReservoirCellularAutomata],
    authors = "Francesco Martinuzzi",
    sitename = "ReservoirCellularAutomata.jl",
    format = Documenter.HTML(;
        canonical = "https://MartinuzziFrancesco.github.io/ReservoirCellularAutomata.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(;
    repo = "github.com/MartinuzziFrancesco/ReservoirCellularAutomata.jl",
    devbranch = "main"
)
