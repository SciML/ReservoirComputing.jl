using Documenter, ReservoirComputing

makedocs(
    modules=[ReservoirComputing],
    clean=true,doctest=false,
    sitename = "ReservoirComputing.jl",
    pages = [
        "ReservoirComputing.jl" => "index.md",
        "General Settings" => Any[
            "Changing Training Algorithms" => "general/different_training.md",
            "Altering States" => "general/states_variation.md",
            "Generative vs Predictive" => "general/predictive_generative.md",            
        ],
        "Echo State Network Tutorials" => Any[
            "Lorenz System Forecasting"=>"esn_tutorials/lorenz_basic.md",
            "Using Different Layers" => "esn_tutorials/change_layers.md",
            #"Multiple Activation Function ESN" => "esn_tutorials/dafesn.md",
            #"Gated Echo State Networks" => "esn_tutorials/gruesn.md",
            "Using Different Reservoir Drivers" => "esn_tutorials/different_drivers.md",
            "Hybrid Echo State Networks" => "esn_tutorials/hybrid.md",
            ],
        #"ReCA" => Any[

        #]},
        "API Documentation"=>Any[
            "Training Algorithms" => "api/training.md",
            "States Modifications" => "api/states.md",
            "Prediction Types" => "api/predict.md",
            "Echo State Networks" => "api/esn.md",
            "ESN Layers" => "api/esn_layers.md",
            "ESN Drivers" => "api/esn_drivers.md",
            #"ReCA" => "api/reca.md"
            ]
        ]
    )

deploydocs(
   repo = "github.com/SciML/ReservoirComputing.jl.git";
   push_preview = true
)
