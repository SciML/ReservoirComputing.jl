struct DeepESN{I, S, N, T, O, M, B, ST, W, IS} <: AbstractEchoStateNetwork
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
    DeepESN(train_data, in_size, res_size; kwargs...)

Constructs a Deep Echo State Network (ESN) model for
processing sequential data through a layered architecture of reservoirs.
This constructor allows for the creation of a deep learning model that
benefits from the dynamic memory and temporal processing capabilities of ESNs,
enhanced by the depth provided by multiple reservoir layers.

# Parameters

  - `train_data`: The training dataset used for the ESN.
    This should be structured as sequential data where applicable.
  - `in_size`: The size of the input layer, i.e., the number of
    input units to the ESN.
  - `res_size`: The size of each reservoir, i.e., the number of neurons
    in each hidden layer of the ESN.

# Optional Keyword Arguments

  - `depth`: The number of reservoir layers in the Deep ESN. Default is 2.
  - `input_layer`: A function or an array of functions to initialize the input
    matrices for each layer. Default is `scaled_rand` for each layer.
  - `bias`: A function or an array of functions to initialize the bias vectors
    for each layer. Default is `zeros32` for each layer.
  - `reservoir`: A function or an array of functions to initialize the reservoir
    matrices for each layer. Default is `rand_sparse` for each layer.
  - `reservoir_driver`: The driving system for the reservoir.
    Default is an RNN model.
  - `nla_type`: The type of non-linear activation used in the reservoir.
    Default is `NLADefault()`.
  - `states_type`: Defines the type of states used in the ESN
    (e.g., standard states). Default is `StandardStates()`.
  - `washout`: The number of initial timesteps to be discarded
    in the ESN's training phase. Default is 0.
  - `rng`: Random number generator used for initializing weights.
    Default is `Utils.default_rng()`.
  - `matrix_type`: The type of matrix used for storing the training data.
    Default is inferred from `train_data`.

# Example

```julia
train_data = rand(Float32, 3, 100)

# Create a DeepESN with specific parameters
deepESN = DeepESN(train_data, 3, 100; depth = 3, washout = 100)
```
"""
function DeepESN(train_data::AbstractArray, in_size::Int, res_size::Int; depth::Int = 2,
        input_layer = fill(scaled_rand, depth), bias = fill(zeros32, depth),
        reservoir = fill(rand_sparse, depth), reservoir_driver::AbstractDriver = RNN(),
        nla_type::NonLinearAlgorithm = NLADefault(),
        states_type::AbstractStates = StandardStates(), washout::Int = 0,
        rng::AbstractRNG = Utils.default_rng(), matrix_type = typeof(train_data))
    if states_type isa AbstractPaddedStates
        in_size = size(train_data, 1) + 1
        train_data = vcat(adapt(matrix_type, ones(1, size(train_data, 2))),
            train_data)
    end

    T = eltype(train_data)
    reservoir_matrix = [reservoir[i](rng, T, res_size, res_size) for i in 1:depth]
    input_matrix = [i == 1 ? input_layer[i](rng, T, res_size, in_size) :
                    input_layer[i](rng, T, res_size, res_size) for i in 1:depth]
    bias_vector = [bias[i](rng, res_size) for i in 1:depth]
    inner_res_driver = reservoir_driver_params(reservoir_driver, res_size, in_size)
    states = create_states(inner_res_driver, train_data, washout, reservoir_matrix,
        input_matrix, bias_vector)
    train_data = train_data[:, (washout + 1):end]

    return DeepESN(res_size, train_data, nla_type, input_matrix,
        inner_res_driver, reservoir_matrix, bias_vector, states_type, washout,
        states)
end
