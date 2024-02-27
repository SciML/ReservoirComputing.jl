abstract type AbstractEchoStateNetwork <: AbstractReservoirComputer end
struct ESN{I, S, V, N, T, O, M, B, ST, W, IS} <: AbstractEchoStateNetwork
    res_size::I
    train_data::S
    variation::V
    nla_type::N
    input_matrix::T
    reservoir_driver::O
    reservoir_matrix::M
    bias_vector::B
    states_type::ST
    washout::W
    states::IS
end

"""
    Default()

The `Default` struct specifies the use of the standard model in Echo State Networks (ESNs).
It requires no parameters and is used when no specific variations or customizations of the ESN model are needed.
This struct is ideal for straightforward applications where the default ESN settings are sufficient.
"""
struct Default <: AbstractVariation end
struct Hybrid{T, K, O, I, S, D} <: AbstractVariation
    prior_model::T
    u0::K
    tspan::O
    dt::I
    datasize::S
    model_data::D
end

"""
    Hybrid(prior_model, u0, tspan, datasize)

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
function Hybrid(prior_model, u0, tspan, datasize)
    trange = collect(range(tspan[1], tspan[2], length = datasize))
    dt = trange[2] - trange[1]
    tsteps = push!(trange, dt + trange[end])
    tspan_new = (tspan[1], dt + tspan[2])
    model_data = prior_model(u0, tspan_new, tsteps)
    return Hybrid(prior_model, u0, tspan, dt, datasize, model_data)
end

"""
    ESN(train_data; kwargs...) -> ESN

Creates an Echo State Network (ESN) using specified parameters and training data, suitable for various machine learning tasks.

# Parameters

  - `train_data`: Matrix of training data (columns as time steps, rows as features).
  - `variation`: Variation of ESN (default: `Default()`).
  - `input_layer`: Input layer of ESN (default: `DenseLayer()`).
  - `reservoir`: Reservoir of the ESN (default: `RandSparseReservoir(100)`).
  - `bias`: Bias vector for each time step (default: `NullLayer()`).
  - `reservoir_driver`: Mechanism for evolving reservoir states (default: `RNN()`).
  - `nla_type`: Non-linear activation type (default: `NLADefault()`).
  - `states_type`: Format for storing states (default: `StandardStates()`).
  - `washout`: Initial time steps to discard (default: `0`).
  - `matrix_type`: Type of matrices used internally (default: type of `train_data`).

# Returns

  - An initialized ESN instance with specified parameters.

# Examples

```julia
using ReservoirComputing

train_data = rand(10, 100)  # 10 features, 100 time steps

esn = ESN(train_data, reservoir = RandSparseReservoir(200), washout = 10)
```
"""
function ESN(train_data;
        variation = Default(),
        input_layer = DenseLayer(),
        reservoir = RandSparseReservoir(100),
        bias = NullLayer(),
        reservoir_driver = RNN(),
        nla_type = NLADefault(),
        states_type = StandardStates(),
        washout = 0,
        matrix_type = typeof(train_data))
    if variation isa Hybrid
        train_data = vcat(train_data, variation.model_data[:, 1:(end - 1)])
    end

    if states_type isa AbstractPaddedStates
        in_size = size(train_data, 1) + 1
        train_data = vcat(Adapt.adapt(matrix_type, ones(1, size(train_data, 2))),
            train_data)
    else
        in_size = size(train_data, 1)
    end

    input_matrix, reservoir_matrix, bias_vector, res_size = obtain_layers(in_size,
        input_layer,
        reservoir, bias;
        matrix_type = matrix_type)

    inner_res_driver = reservoir_driver_params(reservoir_driver, res_size, in_size)
    states = create_states(inner_res_driver, train_data, washout, reservoir_matrix,
        input_matrix, bias_vector)
    train_data = train_data[:, (washout + 1):end]

    ESN(sum(res_size), train_data, variation, nla_type, input_matrix,
        inner_res_driver, reservoir_matrix, bias_vector, states_type, washout,
        states)
end

#shallow esn construction
function obtain_layers(in_size,
        input_layer,
        reservoir,
        bias;
        matrix_type = Matrix{Float64})
    input_res_size = get_ressize(reservoir)
    input_matrix = create_layer(input_layer, input_res_size, in_size,
        matrix_type = matrix_type)
    res_size = size(input_matrix, 1) #WeightedInput actually changes the res size
    reservoir_matrix = create_reservoir(reservoir, res_size, matrix_type = matrix_type)
    @assert size(reservoir_matrix, 1) == res_size
    bias_vector = create_layer(bias, res_size, 1, matrix_type = matrix_type)
    return input_matrix, reservoir_matrix, bias_vector, res_size
end

#deep esn construction
#there is a bug going on with WeightedLayer in this construction.
#it works for eny other though
function obtain_layers(in_size,
        input_layer,
        reservoir::Vector,
        bias;
        matrix_type = Matrix{Float64})
    esn_depth = length(reservoir)
    input_res_sizes = [get_ressize(reservoir[i]) for i in 1:esn_depth]
    in_sizes = zeros(Int, esn_depth)
    in_sizes[2:end] = input_res_sizes[1:(end - 1)]
    in_sizes[1] = in_size

    if input_layer isa Array
        input_matrix = [create_layer(input_layer[j], input_res_sizes[j], in_sizes[j],
                            matrix_type = matrix_type) for j in 1:esn_depth]
    else
        _input_layer = fill(input_layer, esn_depth)
        input_matrix = [create_layer(_input_layer[k], input_res_sizes[k], in_sizes[k],
                            matrix_type = matrix_type) for k in 1:esn_depth]
    end

    res_sizes = [get_ressize(input_matrix[j]) for j in 1:esn_depth]
    reservoir_matrix = [create_reservoir(reservoir[k], res_sizes[k],
                            matrix_type = matrix_type) for k in 1:esn_depth]

    if bias isa Array
        bias_vector = [create_layer(bias[j], res_sizes[j], 1, matrix_type = matrix_type)
                       for j in 1:esn_depth]
    else
        _bias = fill(bias, esn_depth)
        bias_vector = [create_layer(_bias[k], res_sizes[k], 1, matrix_type = matrix_type)
                       for k in 1:esn_depth]
    end

    return input_matrix, reservoir_matrix, bias_vector, res_sizes
end

function (esn::ESN)(prediction::AbstractPrediction,
        output_layer::AbstractOutputLayer;
        last_state = esn.states[:, [end]],
        kwargs...)
    variation = esn.variation
    pred_len = prediction.prediction_len

    if variation isa Hybrid
        model = variation.prior_model
        predict_tsteps = [variation.tspan[2] + variation.dt]
        [append!(predict_tsteps, predict_tsteps[end] + variation.dt) for i in 1:pred_len]
        tspan_new = (variation.tspan[2] + variation.dt, predict_tsteps[end])
        u0 = variation.model_data[:, end]
        model_pred_data = model(u0, tspan_new, predict_tsteps)[:, 2:end]
        return obtain_esn_prediction(esn, prediction, last_state, output_layer,
            model_pred_data;
            kwargs...)
    else
        return obtain_esn_prediction(esn, prediction, last_state, output_layer;
            kwargs...)
    end
end

#training dispatch on esn
"""
    train(esn::AbstractEchoStateNetwork, target_data, training_method = StandardRidge(0.0))

Trains an Echo State Network (ESN) using the provided target data and a specified training method.

# Parameters

  - `esn::AbstractEchoStateNetwork`: The ESN instance to be trained.
  - `target_data`: Supervised training data for the ESN.
  - `training_method`: The method for training the ESN (default: `StandardRidge(0.0)`).

# Returns

  - The trained ESN model. Its type and structure depend on `training_method` and the ESN's implementation.

# Returns

The trained ESN model. The exact type and structure of the return value depends on the
`training_method` and the specific ESN implementation.

```julia
using ReservoirComputing

# Initialize an ESN instance and target data
esn = ESN(train_data, reservoir = RandSparseReservoir(200), washout = 10)
target_data = rand(size(train_data, 2))

# Train the ESN using the default training method
trained_esn = train(esn, target_data)

# Train the ESN using a custom training method
trained_esn = train(esn, target_data, training_method = StandardRidge(1.0))
```

# Notes

  - When using a `Hybrid` variation, the function extends the state matrix with data from the
    physical model included in the `variation`.
  - The training is handled by a lower-level `_train` function which takes the new state matrix
    and performs the actual training using the specified `training_method`.
"""
function train(esn::AbstractEchoStateNetwork,
        target_data,
        training_method = StandardRidge(0.0))
    variation = esn.variation

    if esn.variation isa Hybrid
        states = vcat(esn.states, esn.variation.model_data[:, 2:end])
    else
        states = esn.states
    end
    states_new = esn.states_type(esn.nla_type, states, esn.train_data[:, 1:end])

    return _train(states_new, target_data, training_method)
end

function pad_esnstate(variation::Hybrid, states_type, x_pad, x, model_prediction_data)
    x_tmp = vcat(x, model_prediction_data)
    x_pad = pad_state!(states_type, x_pad, x_tmp)
end

function pad_esnstate!(variation, states_type, x_pad, x, args...)
    x_pad = pad_state!(states_type, x_pad, x)
end

