using Documenter, ReservoirComputing

makedocs(
    modules=[ReservoirComputing],
    clean=true,doctest=false,
    sitename = "ReservoirComputing.jl",
    pages = [
        "ReservoirComputing.jl" => "index.md",
        "Echo State Network Tutorials" => Any[
            "Lorenz System Forecasting"=>"esn_tutorials/lorenz_basic.md",
            #"Using different layers"=>"esn_tutorials/layers.md",
            #"Using different linear methods"=>"esn_tutorials/linear.md",
            #"Double Activation Function ESN"=>"esn_tutorials/dafesn.md",
             #"SVESM"=>"examples/svesm.md",
            #"ESGP"=>"esn_tutorials/esgp.md"
            
            ],
        "API Documentation"=>Any[
            "Training Algorithms" => "api/training.md",
            "States Modifications" => "api/states.md",
            "Prediction Types" => "api/predict.md",
            "Echo State Networks" => "api/esn.md",
            "ESN Layers" => "api/esn_layers.md",
            "ESN Drivers" => "api/esn_drivers.md",
            #"States Types"=>
            ]
        ]
    )

deploydocs(
   repo = "github.com/SciML/ReservoirComputing.jl.git";
   push_preview = true
)
