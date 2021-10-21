
mutable struct ESN{I,S,N,T,O,M,IS} <: AbstractReservoirComputer
    res_size::I
    train_data::S
    nla_type::N
    input_matrix::T
    reservoir_driver::O 
    reservoir_matrix::M
    states::IS
    extended_states::Bool
end

"""
    ESN(W::AbstractArray{T}, train_data::AbstractArray{T}, W_in::AbstractArray{T}
    [, activation::Any, alpha::T, nla_type::NonLinearAlgorithm, extended_states::Bool])

Build an ESN struct given the input and reservoir matrices.
"""
function ESN(input_res_size, train_data;
             input_init = WeightedInput(),
             reservoir_init = RandReservoir(),
             reservoir_driver = RNN(),
             nla_type = NLADefault(),
             extended_states = false)

    in_size = size(train_data, 1)
    input_matrix = create_layer(input_res_size, in_size, input_init)
    res_size = size(input_matrix, 1) #WeightedInput actually changes the res size
    reservoir_matrix = create_reservoir(res_size, reservoir_init)
    states = create_states(reservoir_driver, train_data, extended_states, reservoir_matrix, input_matrix)

    ESN(res_size, train_data, nla_type, input_matrix, reservoir_driver, 
        reservoir_matrix, states, extended_states)
end

struct Autonomous{O,T} <: AbstractPrediction
    output_layer::O
    prediction_len::T
end

function Autonomous(output_layer; prediction_len=100)
    Autonomous(output_layer, prediction_len)
end

struct Direct{O,T} <: AbstractPrediction
    output_layer::O
    prediction_data::T
end

function (esn::ESN)(aut::Autonomous)

    output = zeros(size(aut.output_layer, 1), aut.prediction_len) #better way to check size output?
    x = esn.states[:, end] 

    for i=1:aut.prediction_len
        x_new = nla(esn.nla_type, x)
        out = (aut.output_layer*x_new)
        output[:, i] = out
        esn.extended_states ? x = vcat(next_state(esn.reservoir_driver, x[1:esn.res_size], out, esn.reservoir_matrix, 
        esn.input_matrix), out) : x = next_state(esn.reservoir_driver, x, out, esn.reservoir_matrix, esn.input_matrix)
    end
    output
end

function (esn::ESN)(direct::Direct)

    prediction_len = size(direct.prediction_data, 2)
    output = zeros(size(direct.output_layer, 1), prediction_len)
    x = esn.states[:, end] #x = zeros(size(esn.states,2))

    for i=1:prediction_len
        esn.extended_states ? x = vcat(next_state(esn.reservoir_driver, x[1:esn.res_size], direct.prediction_data[:,i], 
        esn.reservoir_matrix, esn.input_matrix), direct.prediction_data[:,i]) : x = next_state(esn.reservoir_driver, x, 
        direct.prediction_data[:,i], esn.reservoir_matrix, esn.input_matrix)
        x_new = nla(esn.nla_type, x)
        out = (direct.output_layer*x_new)
        output[:, i] = out    
    end
    output
end

