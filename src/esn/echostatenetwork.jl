abstract type AbstractEchoStateNetwork <: AbstractReservoirComputer end
struct ESN{I,S,V,N,T,O,M,ST,IS} <: AbstractEchoStateNetwork
    res_size::I
    train_data::S
    variation::V
    nla_type::N
    input_matrix::T
    reservoir_driver::O 
    reservoir_matrix::M
    states_type::ST
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
             states_type = StandardStates())

    variation isa Hybrid ? train_data = vcat(train_data, variation.model_data[:, 1:end-1]) : nothing
    in_size = size(train_data, 1)
    input_matrix = create_layer(input_init, input_res_size, in_size)
    res_size = size(input_matrix, 1) #WeightedInput actually changes the res size
    reservoir_matrix = create_reservoir(reservoir_init, res_size)
    inner_reservoir_driver = reservoir_driver_params(reservoir_driver, res_size, in_size)
    states = create_states(inner_reservoir_driver, train_data, reservoir_matrix, input_matrix)

    ESN(res_size, train_data, variation, nla_type, input_matrix, inner_reservoir_driver, 
        reservoir_matrix, states_type, states)
end


function (esn::ESN)(prediction,
    output_layer::AbstractOutputLayer;
    initial_conditions=output_layer.last_value,
    last_state=esn.states[:, end])

    variation = esn.variation

    if variation isa Hybrid
        predict_tsteps = [variation.tspan[2]+variation.dt]
        [append!(predict_tsteps, predict_tsteps[end]+variation.dt) for i in 1:prediction.prediction_len]
        tspan_new = (variation.tspan[2]+variation.dt, predict_tsteps[end])
        u0 = variation.model_data[:, end]
        model_prediction_data = variation.prior_model(u0, tspan_new, predict_tsteps)[:, 2:end]
        return obtain_prediction(esn, prediction, last_state, output_layer, model_prediction_data; initial_conditions=initial_conditions)
    else 
        return obtain_prediction(esn, prediction, last_state, output_layer; initial_conditions=initial_conditions)
    end
end

#training dispatch on esn
function train(esn::AbstractEchoStateNetwork, target_data, training_method=StandardRidge(0.0))

    esn.variation isa Hybrid ? states = vcat(esn.states, esn.variation.model_data[:, 2:end]) : states=esn.states
    states_new = esn.states_type(esn.nla_type, states, esn.train_data[:,1:end])

    _train(states_new, target_data, training_method)
end

#prediction dispatch on esn 
function next_state_prediction!(esn::ESN, x, out, i, args...)
    _variation_prediction!(esn.variation, esn, x, out, i, args...)
end

#dispatch the prediction on the esn variation
function _variation_prediction!(variation, esn, x, out, i, args...)
    x = next_state(esn.reservoir_driver, x[1:esn.res_size], out, esn.reservoir_matrix, esn.input_matrix)
    x_new = esn.states_type(esn.nla_type, x, out)
    x, x_new
end

function _variation_prediction!(variation::Hybrid, esn, x, out, i, model_prediction_data)
    out_tmp = vcat(out, model_prediction_data[:,i])
    x = next_state(esn.reservoir_driver, x[1:esn.res_size], out_tmp, esn.reservoir_matrix, esn.input_matrix)
    x_tmp = vcat(x, model_prediction_data[:,i])
    x_new = esn.states_type(esn.nla_type, x_tmp, out_tmp)
    x, x_new
end