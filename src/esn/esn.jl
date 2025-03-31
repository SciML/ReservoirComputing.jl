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

const AbstractDriver = Union{AbstractReservoirDriver, GRU}

"""
    ESN(train_data; kwargs...) -> ESN

Creates an Echo State Network (ESN).

# Arguments

  - `train_data`: Matrix of training data `num_features x time_steps`.
  - `variation`: Variation of ESN (default: `Default()`).
  - `input_layer`: Input layer of ESN.
  - `reservoir`: Reservoir of the ESN.
  - `bias`: Bias vector for each time step.
  - `rng`: Random number generator used for initializing weights.
    Default is `Utils.default_rng()`.
  - `reservoir_driver`: Mechanism for evolving reservoir states (default: `RNN()`).
  - `nla_type`: Non-linear activation type (default: `NLADefault()`).
  - `states_type`: Format for storing states (default: `StandardStates()`).
  - `washout`: Initial time steps to discard (default: `0`).
  - `matrix_type`: Type of matrices used internally (default: type of `train_data`).

# Examples

```jldoctest
julia> train_data = rand(Float32, 10, 100)  # 10 features, 100 time steps
10×100 Matrix{Float32}:
 0.567676   0.154756  0.584611  0.294015   …  0.573946    0.894333    0.429133
 0.327073   0.729521  0.804667  0.263944      0.559342    0.020167    0.897862
 0.453606   0.800058  0.568311  0.749441      0.0713146   0.464795    0.532854
 0.0173253  0.536959  0.722116  0.910328      0.00224048  0.00202501  0.631075
 0.366744   0.119761  0.100593  0.125122      0.700562    0.675474    0.102947
 0.539737   0.768351  0.54681   0.648672   …  0.256738    0.223784    0.94327
 0.558099   0.42676   0.1948    0.735625      0.0989234   0.119342    0.624182
 0.0603135  0.929999  0.263439  0.0372732     0.066125    0.332769    0.25562
 0.4463     0.334423  0.444679  0.311695      0.0494497   0.27171     0.214925
 0.987182   0.898593  0.295241  0.233098      0.789699    0.453692    0.759205

julia> esn = ESN(train_data, 10, 300; washout = 10)
ESN(10 => 300)
```
"""
function ESN(train_data::AbstractArray, in_size::Int, res_size::Int;
        input_layer = scaled_rand, reservoir = rand_sparse, bias = zeros32,
        reservoir_driver::AbstractDriver = RNN(),
        nla_type::NonLinearAlgorithm = NLADefault(),
        states_type::AbstractStates = StandardStates(),
        washout::Int = 0, rng::AbstractRNG = Utils.default_rng(),
        matrix_type = typeof(train_data))
    if states_type isa AbstractPaddedStates
        in_size = size(train_data, 1) + 1
        train_data = vcat(adapt(matrix_type, ones(1, size(train_data, 2))),
            train_data)
    end

    T = eltype(train_data)
    reservoir_matrix = reservoir(rng, T, res_size, res_size)
    input_matrix = input_layer(rng, T, res_size, in_size)
    bias_vector = bias(rng, res_size)
    inner_res_driver = reservoir_driver_params(reservoir_driver, res_size, in_size)
    states = create_states(inner_res_driver, train_data, washout, reservoir_matrix,
        input_matrix, bias_vector)
    train_data = train_data[:, (washout + 1):end]

    return ESN(res_size, train_data, nla_type, input_matrix,
        inner_res_driver, reservoir_matrix, bias_vector, states_type, washout,
        states)
end

function (esn::AbstractEchoStateNetwork)(prediction::AbstractPrediction,
        output_layer::AbstractOutputLayer; last_state = esn.states[:, [end]],
        kwargs...)
    return obtain_esn_prediction(esn, prediction, last_state, output_layer;
        kwargs...)
end

function Base.show(io::IO, esn::ESN)
    print(io, "ESN(", size(esn.train_data, 1), " => ", size(esn.reservoir_matrix, 1), ")")
end

#training dispatch on esn
"""
    train(esn::AbstractEchoStateNetwork, target_data, training_method = StandardRidge(0.0))

Trains an Echo State Network (ESN) using the provided target data and a specified training method.

# Parameters

  - `esn::AbstractEchoStateNetwork`: The ESN instance to be trained.
  - `target_data`: Supervised training data for the ESN.
  - `training_method`: The method for training the ESN (default: `StandardRidge(0.0)`).

# Example

```jldoctest
julia> train_data = rand(Float32, 10, 100)  # 10 features, 100 time steps
10×100 Matrix{Float32}:
 0.11437   0.425367  0.585867   0.34078   …  0.0531493  0.761425  0.883164
 0.301373  0.497806  0.279603   0.802417     0.49873    0.270156  0.333333
 0.135224  0.660179  0.394233   0.512753     0.901221   0.784377  0.687691
 0.510203  0.877234  0.614245   0.978405     0.332775   0.768826  0.527077
 0.955027  0.398322  0.312156   0.981938     0.473357   0.156704  0.476101
 0.353024  0.997632  0.164328   0.470783  …  0.745613   0.85797   0.465201
 0.966044  0.194299  0.599167   0.040475     0.0996013  0.325959  0.770103
 0.292068  0.495138  0.481299   0.214566     0.819573   0.155951  0.227168
 0.133498  0.451058  0.0761995  0.90421      0.994212   0.332164  0.545112
 0.214467  0.791524  0.124105   0.951805     0.947166   0.954244  0.889733

julia> esn = ESN(train_data, 10, 300; washout = 10)
ESN(10 => 300)

julia> output_layer = train(esn, rand(Float32, 3, 90))
OutputLayer successfully trained with output size: 3
```
"""
function train(esn::AbstractEchoStateNetwork, target_data::AbstractArray,
        training_method = StandardRidge(); kwargs...)
    states_new = esn.states_type(esn.nla_type, esn.states, esn.train_data[:, 1:end])
    return train(training_method, states_new, target_data; kwargs...)
end

#function pad_esnstate(variation::Hybrid, states_type, x_pad, x, model_prediction_data)
#    x_tmp = vcat(x, model_prediction_data)
#    x_pad = pad_state!(states_type, x_pad, x_tmp)
#end

#function pad_esnstate!(variation, states_type, x_pad, x, args...)
#    x_pad = pad_state!(states_type, x_pad, x)
#end
