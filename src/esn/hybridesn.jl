struct HybridESN{I, S, V, N, T, O, M, B, ST, W, IS} <: AbstractEchoStateNetwork
    res_size::I
    train_data::S
    model::V
    nla_type::N
    input_matrix::T
    reservoir_driver::O
    reservoir_matrix::M
    bias_vector::B
    states_type::ST
    washout::W
    states::IS
end

struct KnowledgeModel{T, K, O, I, S, D}
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
function KnowledgeModel(prior_model, u0, tspan, datasize)
    trange = collect(range(tspan[1], tspan[2], length = datasize))
    dt = trange[2] - trange[1]
    tsteps = push!(trange, dt + trange[end])
    tspan_new = (tspan[1], dt + tspan[2])
    model_data = prior_model(u0, tspan_new, tsteps)
    return Hybrid(prior_model, u0, tspan, dt, datasize, model_data)
end

function (hesn::HybridESN)(prediction::AbstractPrediction,
    output_layer::AbstractOutputLayer;
    last_state = esn.states[:, [end]],
    kwargs...)

    pred_len = prediction.prediction_len

    model = variation.prior_model
    predict_tsteps = [variation.tspan[2] + variation.dt]
    [append!(predict_tsteps, predict_tsteps[end] + variation.dt) for i in 1:pred_len]
    tspan_new = (variation.tspan[2] + variation.dt, predict_tsteps[end])
    u0 = variation.model_data[:, end]
    model_pred_data = model(u0, tspan_new, predict_tsteps)[:, 2:end]

    return obtain_esn_prediction(esn, prediction, last_state, output_layer,
        model_pred_data;
        kwargs...)
end

function train(hesn::HybridESN,
    target_data,
    training_method = StandardRidge(0.0))

    states = vcat(esn.states, esn.variation.model_data[:, 2:end])
    states_new = esn.states_type(esn.nla_type, states, esn.train_data[:, 1:end])

    return _train(states_new, target_data, training_method)
end