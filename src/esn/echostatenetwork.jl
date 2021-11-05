
abstract type AbstractEchoStateNetwork <: AbstractReservoirComputer end
struct ESN{I,S,V,N,T,O,M,IS} <: AbstractEchoStateNetwork
    res_size::I
    train_data::S
    variation::V
    nla_type::N
    input_matrix::T
    reservoir_driver::O 
    reservoir_matrix::M
    extended_states::Bool
    states::IS
end

"""
    Default()

Given as input to ```variation``` return a standard model of the ESN. No parameters are needed.
"""
struct Default <: AbstractVariation end
struct Hybrid{T,K,O,I,S,D} <: AbstractVariation
    prior_model::T
    u0::K
    tspan::O
    dt::I
    datasize::S
    model_data::D
end

"""
    Hybrid(prior_model, u0, tspan, datasize)

Given the model parameters this return an ```Hybrid``` ```variation``` of the ESN. This entails a different training 
and prediction. Construction taken from [1].

[1] Jaideep Pathak et al. "Hybrid Forecasting of Chaotic Processes: Using Machine Learning in Conjunction with a Knowledge-Based Model" (2018)
"""
function Hybrid(prior_model, u0, tspan, datasize)
    trange = collect(range(tspan[1], tspan[2], length = datasize))
    dt = trange[2]-trange[1]
    tsteps = push!(trange, dt + trange[end])
    tspan_new = (tspan[1], dt+tspan[2])
    model_data = prior_model(u0, tspan_new, tsteps)
    Hybrid(prior_model, u0, tspan, dt, datasize, model_data)
end

"""
    ESN()
"""
function ESN(input_res_size, train_data;
             variation = Default(),
             input_init = WeightedLayer(),
             reservoir_init = RandSparseReservoir(),
             reservoir_driver = RNN(),
             nla_type = NLADefault(),
             extended_states = false)

    variation isa Hybrid ? train_data = vcat(train_data, variation.model_data[:, 1:end-1]) : nothing
    in_size = size(train_data, 1)
    input_matrix = create_layer(input_res_size, in_size, input_init)
    res_size = size(input_matrix, 1) #WeightedInput actually changes the res size
    reservoir_matrix = create_reservoir(res_size, reservoir_init)
    inner_reservoir_driver = reservoir_driver_params(reservoir_driver, res_size, in_size)
    states = create_states(inner_reservoir_driver, variation, train_data, reservoir_matrix, input_matrix)

    ESN(res_size, train_data, variation, nla_type, input_matrix, inner_reservoir_driver, 
        reservoir_matrix, extended_states, states)
end

function (esn::ESN)(aut::Autonomous, output_layer::AbstractOutputLayer)

    output = obtain_autonomous_prediction(esn, output_layer, aut.prediction_len, 
                                          output_layer.training_method, esn.variation) #dispatch on prediction type -> just obtain_prediction()
    output
end

function (esn::ESN)(direct::Direct, output_layer::AbstractOutputLayer)

    output = obtain_direct_prediction(esn, output_layer, direct.prediction_data, 
                                      output_layer.training_method, esn.variation) 
    output
end

function (esn::ESN)(fitted::Fitted, output_layer::AbstractOutputLayer)
    if fitted.type == Direct
        output = obtain_direct_prediction(esn, output_layer, esn.train_data, 
                                          output_layer.training_method, esn.variation)
    elseif fitted.type == Autonomous #need to change actual starting state I think
        prediction_len = size(esn.states, 2)
        output = obtain_autonomous_prediction(esn, output_layer, prediction_len, 
                                              output_layer.training_method, esn.variation)
    end
    output
end
