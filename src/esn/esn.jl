abstract type AbstractEchoStateNetwork <: AbstractReservoirComputer end
struct ESN{I, S, N, T, O, M, B, ST, W, IS} <: AbstractEchoStateNetwork
    res_size::I
    train_data::S
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

esn = ESN(train_data, reservoir=RandSparseReservoir(200), washout=10)
```
"""
function ESN(train_data,
        in_size::Int,
        res_size::Int;
        input_layer = scaled_rand,
        reservoir = rand_sparse,
        bias = zeros64,
        reservoir_driver = RNN(),
        nla_type = NLADefault(),
        states_type = StandardStates(),
        washout = 0,
        rng = _default_rng(),
        T = Float32,
        matrix_type = typeof(train_data))
    if states_type isa AbstractPaddedStates
        in_size = size(train_data, 1) + 1
        train_data = vcat(Adapt.adapt(matrix_type, ones(1, size(train_data, 2))),
            train_data)
    end

    reservoir_matrix = reservoir(rng, T, res_size, res_size)
    input_matrix = input_layer(rng, T, res_size, in_size)
    bias_vector = bias(rng, res_size)
    inner_res_driver = reservoir_driver_params(reservoir_driver, res_size, in_size)
    states = create_states(inner_res_driver, train_data, washout, reservoir_matrix,
        input_matrix, bias_vector)
    train_data = train_data[:, (washout + 1):end]

    ESN(res_size, train_data, nla_type, input_matrix,
        inner_res_driver, reservoir_matrix, bias_vector, states_type, washout,
        states)
end

function (esn::ESN)(prediction::AbstractPrediction,
        output_layer::AbstractOutputLayer;
        last_state = esn.states[:, [end]],
        kwargs...)
    pred_len = prediction.prediction_len

    return obtain_esn_prediction(esn, prediction, last_state, output_layer;
        kwargs...)
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
esn = ESN(train_data, reservoir=RandSparseReservoir(200), washout=10)
target_data = rand(size(train_data, 2))

# Train the ESN using the default training method
trained_esn = train(esn, target_data)

# Train the ESN using a custom training method
trained_esn = train(esn, target_data, training_method=StandardRidge(1.0))
```

# Notes
- When using a `Hybrid` variation, the function extends the state matrix with data from the
    physical model included in the `variation`.
- The training is handled by a lower-level `_train` function which takes the new state matrix
    and performs the actual training using the specified `training_method`.
"""
function train(esn::ESN,
        target_data,
        training_method = StandardRidge(0.0))
    states_new = esn.states_type(esn.nla_type, esn.states, esn.train_data[:, 1:end])

    return _train(states_new, target_data, training_method)
end

#function pad_esnstate(variation::Hybrid, states_type, x_pad, x, model_prediction_data)
#    x_tmp = vcat(x, model_prediction_data)
#    x_pad = pad_state!(states_type, x_pad, x_tmp)
#end

#function pad_esnstate!(variation, states_type, x_pad, x, args...)
#    x_pad = pad_state!(states_type, x_pad, x)
#end
