struct HybridESN{I, S, V, N, T, O, M, B, ST, W, IS} <: AbstractEchoStateNetwork
    res_size::I
    train_data::S
    model::V
    nla_type::N
    input_matrix::T
    reservoir_driver::O
    reservoir_matrix::M
    bias_vector::B
    states_type::ST
    washout::W
    states::IS
end

struct KnowledgeModel{T, K, O, I, S, D}
    prior_model::T
    u0::K
    tspan::O
    dt::I
    datasize::S
    model_data::D
end

"""
KnowledgeModel(prior_model, u0, tspan, datasize)

Constructs a `Hybrid` variation of Echo State Networks (ESNs) integrating a knowledge-based model
(`prior_model`) with ESNs for advanced training and prediction in chaotic systems.

# Parameters

  - `prior_model`: A knowledge-based model function for integration with ESNs.
  - `u0`: Initial conditions for the model.
  - `tspan`: Time span as a tuple, indicating the duration for model operation.
  - `datasize`: The size of the data to be processed.

# Returns

  - A `Hybrid` struct instance representing the combined ESN and knowledge-based model.

This method is effective for chaotic processes as highlighted in [^Pathak].

Reference:

[^Pathak]: Jaideep Pathak et al.
    "Hybrid Forecasting of Chaotic Processes:
    Using Machine Learning in Conjunction with a Knowledge-Based Model" (2018).
"""
function KnowledgeModel(prior_model, u0, tspan, datasize)
    trange = collect(range(tspan[1], tspan[2], length = datasize))
    dt = trange[2] - trange[1]
    tsteps = push!(trange, dt + trange[end])
    tspan_new = (tspan[1], dt + tspan[2])
    model_data = prior_model(u0, tspan_new, tsteps)
    return KnowledgeModel(prior_model, u0, tspan, dt, datasize, model_data)
end

"""
    HybridESN(model, train_data, in_size, res_size; kwargs...)

Construct a Hybrid Echo State Network (ESN) model that integrates
traditional Echo State Networks with a predefined knowledge model for
enhanced performance on chaotic systems or complex datasets. This
constructor allows for the creation of a customized ESN architecture by
specifying the reservoir size, input size, and various other parameters that
influence the network's behavior and learning capacity.

# Parameters

  - `model`: A `KnowledgeModel` instance representing the knowledge-based model
    to be integrated with the ESN.
  - `train_data`: The training dataset used for the ESN. This data can be
    preprocessed or raw data depending on the nature of the problem and the
    preprocessing steps considered.
  - `in_size`: The size of the input layer, i.e., the number of input units
    to the ESN.
  - `res_size`: The size of the reservoir, i.e., the number of neurons in
    the hidden layer of the ESN.

# Optional Keyword Arguments

  - `input_layer`: A function to initialize the input matrix. Default is `scaled_rand`.
  - `reservoir`: A function to initialize the reservoir matrix. Default is `rand_sparse`.
  - `bias`: A function to initialize the bias vector. Default is `zeros64`.
  - `reservoir_driver`: The driving system for the reservoir. Default is an RNN model.
  - `nla_type`: The type of non-linear activation used in the reservoir.
    Default is `NLADefault()`.
  - `states_type`: Defines the type of states used in the ESN (e.g., standard states).
    Default is `StandardStates()`.
  - `washout`: The number of initial timesteps to be discarded in the ESN's training phase.
    Default is 0.
  - `rng`: Random number generator used for initializing weights. Default is the package's
    default random number generator.
  - `T`: The data type for the matrices (e.g., `Float32`). Influences computational
    efficiency and precision.
  - `matrix_type`: The type of matrix used for storing the training data. Default is
    inferred from `train_data`.

# Returns

  - A `HybridESN` instance configured according to the provided parameters and
    suitable for further training and prediction tasks.

# Example

```julia
# Define a KnowledgeModel
km = KnowledgeModel(prior_model_function, u0, (0, 100), 1000)

# Create a HybridESN
hesn = HybridESN(km, train_data, 10, 100; washout = 100)

# Train and predict
train(hesn, target_data)
prediction = hesn(prediction_object, output_layer)
```
"""
function HybridESN(model,
        train_data,
        in_size::Int,
        res_size::Int;
        input_layer = scaled_rand,
        reservoir = rand_sparse,
        bias = zeros64,
        reservoir_driver = RNN(),
        nla_type = NLADefault(),
        states_type = StandardStates(),
        washout = 0,
        rng = WeightInitializers._default_rng(),
        T = Float32,
        matrix_type = typeof(train_data))
    train_data = vcat(train_data, model.model_data[:, 1:(end - 1)])

    if states_type isa AbstractPaddedStates
        in_size = size(train_data, 1) + 1
        train_data = vcat(Adapt.adapt(matrix_type, ones(1, size(train_data, 2))),
            train_data)
    else
        in_size = size(train_data, 1)
    end

    reservoir_matrix = reservoir(rng, T, res_size, res_size)
    #different from ESN, why?
    input_matrix = input_layer(rng, T, res_size, in_size)
    bias_vector = bias(rng, res_size)
    inner_res_driver = reservoir_driver_params(reservoir_driver, res_size, in_size)
    states = create_states(inner_res_driver, train_data, washout, reservoir_matrix,
        input_matrix, bias_vector)
    train_data = train_data[:, (washout + 1):end]

    HybridESN(res_size, train_data, model, nla_type, input_matrix,
        inner_res_driver, reservoir_matrix, bias_vector, states_type, washout,
        states)
end

function (hesn::HybridESN)(prediction::AbstractPrediction,
        output_layer::AbstractOutputLayer;
        last_state = hesn.states[:, [end]],
        kwargs...)
    km = hesn.model
    pred_len = prediction.prediction_len

    model = km.prior_model
    predict_tsteps = [km.tspan[2] + km.dt]
    [append!(predict_tsteps, predict_tsteps[end] + km.dt) for i in 1:pred_len]
    tspan_new = (km.tspan[2] + km.dt, predict_tsteps[end])
    u0 = km.model_data[:, end]
    model_pred_data = model(u0, tspan_new, predict_tsteps)[:, 2:end]

    return obtain_esn_prediction(hesn, prediction, last_state, output_layer,
        model_pred_data;
        kwargs...)
end

function train(hesn::HybridESN,
        target_data,
        training_method = StandardRidge();
        kwargs...)
    states = vcat(hesn.states, hesn.model.model_data[:, 2:end])
    states_new = hesn.states_type(hesn.nla_type, states, hesn.train_data[:, 1:end])

    return train(training_method, states_new, target_data; kwargs...)
end
