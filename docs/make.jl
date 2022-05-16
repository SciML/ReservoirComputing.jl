using Documenter, ReservoirComputing

makedocs(
    modules=[ReservoirComputing],
    clean=true,doctest=false,
    sitename = "ReservoirComputing.jl",
    format = Documenter.HTML(analytics = "UA-90474609-3",
                         assets = ["assets/favicon.ico"],
                         canonical="https://reservoircomputing.sciml.ai/stable/"),
    pages = [
        "ReservoirComputing.jl" => "index.md",
        "General Settings" => Any[
            "Changing Training Algorithms" => "general/different_training.md",
            "Altering States" => "general/states_variation.md",
            "Generative vs Predictive" => "general/predictive_generative.md",            
        ],
        "Echo State Network Tutorials" => Any[
            "Lorenz System Forecasting" => "esn_tutorials/lorenz_basic.md",
            "Mackey-Glass Forecasting on GPU" => "esn_tutorials/mackeyglass_basic.md",
            "Using Different Layers" => "esn_tutorials/change_layers.md",
            "Using Different Reservoir Drivers" => "esn_tutorials/different_drivers.md",
            #"Using Different Training Methods" => "esn_tutorials/different_training.md",
            "Deep Echo State Networks" => "esn_tutorials/deep_esn.md",
            "Hybrid Echo State Networks" => "esn_tutorials/hybrid.md",
            ],
        "Reservoir Computing with Cellular Automata" => "reca_tutorials/reca.md",
        "API Documentation"=>Any[
            "Training Algorithms" => "api/training.md",
            "States Modifications" => "api/states.md",
            "Prediction Types" => "api/predict.md",
            "Echo State Networks" => "api/esn.md",
            "ESN Layers" => "api/esn_layers.md",
            "ESN Drivers" => "api/esn_drivers.md",
            "ReCA" => "api/reca.md"
            ]
        ]
    )

deploydocs(
   repo = "github.com/SciML/ReservoirComputing.jl.git";
   push_preview = true
)
