using Documenter, ReservoirComputing

include("pages.jl")

makedocs(modules = [ReservoirComputing],
         clean = true, doctest = false,
         sitename = "ReservoirComputing.jl",
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://docs.sciml.ai/ReservoirComputing/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/ReservoirComputing.jl.git";
           push_preview = true)
