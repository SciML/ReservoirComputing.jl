
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

function (esn::ESN)(aut::Autonomous, output_layer::AbstractOutputLayer)

    output = obtain_autonomous_prediction(esn, output_layer, aut.prediction_len, 
                                          output_layer.training_method)
    output
end

function (esn::ESN)(direct::Direct, output_layer::AbstractOutputLayer)

    output = obtain_direct_prediction(esn, output_layer, direct.prediction_data, 
                                      output_layer.training_method)
    output
end