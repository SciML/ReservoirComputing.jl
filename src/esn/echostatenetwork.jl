
mutable struct ESN{I,S,N,IL,T,O,R,M,IS,B,OL} <: AbstractEchoStateNetwork
    res_size::I
    train_data::S
    nla_type::N
    input_init::IL
    input_matrix::T
    reservoir_driver::O 
    reservoir_init::R
    reservoir_matrix::M
    states::IS
    extended_states::B
    output_layer::OL
end

"""
    ESN(W::AbstractArray{T}, train_data::AbstractArray{T}, W_in::AbstractArray{T}
    [, activation::Any, alpha::T, nla_type::NonLinearAlgorithm, extended_states::Bool])

Build an ESN struct given the input and reservoir matrices.
"""
function ESN(res_size, train_data;
             input_init = WeightedInput(),
             reservoir_init = RandReservoir(),
             reservoir_driver = RNN(),
             nla_type = NLADefault(),
             extended_states = false)

    in_size = size(train_data, 1)
    reservoir_matrix = create_reservoir(res_size, reservoir_init)
    input_matrix = create_input_layer(res_size, in_size, input_init)
    states = create_states(reservoir_matrix, input_matrix, train_data, 
                           extended_states, reservoir_driver)
    output_layer = zeros(in_size, res_size)

    ESN(res_size, train_data, nla_type, input_init, input_matrix, reservoir_driver, 
        reservoir_init, reservoir_matrix, states, extended_states, output_layer)
end

"""
    ESNpredict(esn::AbstractLeakyESN, predict_len::Int, W_out::AbstractArray{Float64})

Return the prediction for a given length of the constructed ESN struct.
"""
function (esn::ESN)(predict_len)

    output = zeros(Float64, size(esn.output_layer, 1), predict_len)
    x = esn.states[:, end]

    if esn.extended_states == false
        for i=1:predict_len
            x_new = nla(esn.nla_type, x)
            out = (esn.output_layer*x_new)
            output[:, i] = out
            x = next_state(esn.reservoir_driver, esn.reservoir_matrix, esn.input_matrix, x, out)
        end
    elseif esn.extended_states == true
        for i=1:predict_len
            x_new = nla(esn.nla_type, x)
            out = (esn.output_layer*x_new)
            output[:, i] = out
            x = vcat(next_state(esn.reservoir_driver, esn.reservoir_matrix, esn.input_matrix, 
                                x[1:esn.res_size], out), out)
        end
    end
    return output
end