
mutable struct ESN{I,S,N,IL,O,R,S,B,OL} <: AbstractEchoStateNetwork
    res_size::I
    train_data::S
    nla_type::N
    input_init::IL
    reservoir_driver::O 
    reservoir_init::R
    states::S
    extended_states::B
    output_layer::OL
end

"""
    ESN(W::AbstractArray{T}, train_data::AbstractArray{T}, W_in::AbstractArray{T}
    [, activation::Any, alpha::T, nla_type::NonLinearAlgorithm, extended_states::Bool])

Build an ESN struct given the input and reservoir matrices.
"""
function ESN(res_size, train_data;
             reservoir_init = RandReservoir()
             input_init = WeightedInput()
             reservoir_driver = RNN()
             nla_type = NLADefault(),
             extended_states = false)

    in_size = size(train_data, 1)
    reservoir_matrix = create_reservoir(res_size, reservoir_init)
    input_matrix = create_input_layer(res_size, in_size, input_init)
    states = create_states(reservoir_matrix, input_matrix, train_data, extended_states, 
                           nla_type, reservoir_driver)
    output_layer = nothing

    ESN(res_size, train_data, nla_type, input_init, reservoir_driver, reservoir_init,
        states, extended_states, output_layer)
end

"""
    ESNpredict(esn::AbstractLeakyESN, predict_len::Int, W_out::AbstractArray{Float64})

Return the prediction for a given length of the constructed ESN struct.
"""
function (esn::ESN)(predict_len)

    output = zeros(Float64, esn.in_size, predict_len)
    x = esn.states[:, end]

    if esn.extended_states == false
        for i=1:predict_len
            x_new = nla(esn.nla_type, x)
            out = (esn.output_layer*x_new)
            output[:, i] = out
            x = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x, out)
        end
    elseif esn.extended_states == true
        for i=1:predict_len
            x_new = nla(esn.nla_type, x)
            out = (esn.output_layer*x_new)
            output[:, i] = out
            x = vcat(leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, 
                                     x[1:esn.res_size], out), out)
        end
    end
    return output
end