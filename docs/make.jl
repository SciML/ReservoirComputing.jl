using Documenter, ReservoirComputing

makedocs(
    sitename = "ReservoirComputing.jl",
    pages = [
        "ReservoirComputing.jl" => "index.md",
        "Tutorials" => Any[
            "Basics"=>"examples/esn.md",
            "Using different layers"=>"examples/layers.md",
            "Using different linear methods"=>"examples/linear.md",
            "Double Activation Function ESN"=>"examples/dafesn.md",
             #"SVESM"=>"examples/svesm.md",
            "ESGP"=>"examples/esgp.md"
            
            ]
        ]
    )
