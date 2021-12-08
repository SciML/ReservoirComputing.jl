using Documenter, ReservoirComputing

makedocs(
    modules=[ReservoirComputing],
    clean=true,doctest=false,
    sitename = "ReservoirComputing.jl",
    pages = [
        "ReservoirComputing.jl" => "index.md",
        "Echo State Network Tutorials" => Any[
            "Lorenz System Forecasting"=>"esn_tutorials/lorenz_basic.md",
            "Using Different Layers" => "esn_tutorials/change_layers.md",
            "Changing Training Algorithms" => "esn_tutorials/different_training.md",
            "Generative vs Predictive" => "esn_tutorials/predictive_generative.md",
            "Altering ESN States" => "esn_tutorials/states_variation.md",
            "Multiple Activation Function ESN" => "esn_tutorials/dafesn.md",
            "Gated Echo State Networks" => "esn_tutorials/gruesn.md",
            ],
        "API Documentation"=>Any[
            "Training Algorithms" => "api/training.md",
            "States Modifications" => "api/states.md",
            "Prediction Types" => "api/predict.md",
            "Echo State Networks" => "api/esn.md",
            "ESN Layers" => "api/esn_layers.md",
            "ESN Drivers" => "api/esn_drivers.md",
            ]
        ]
    )

deploydocs(
   repo = "github.com/SciML/ReservoirComputing.jl.git";
   push_preview = true
)
