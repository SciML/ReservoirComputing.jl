abstract type AbstractHESN <: AbstractEchoStateNetwork end

struct HESN{T, S<:AbstractArray{T}, I, B, F, N, M, U} <: AbstractHESN
    res_size::I
    in_size::I
    out_size::I
    train_data::S
    physics_model_data::S
    prior_model::M
    u0::U
    tspan::Tuple
    datasize::I
    dt::T
    alpha::T
    nla_type::N
    activation::F
    W::S
    W_in::S
    states::S
    extended_states::B
end

function HESN(W::AbstractArray{T},
        train_data::AbstractArray{T},
        prior_model::Any,
        u0::AbstractArray{T},
        tspan::Tuple,
        datasize::Int,
        W_in::AbstractArray{T};
        activation::Any = tanh,
        alpha::T = 1.0,
        nla_type::NonLinearAlgorithm = NLADefault(),
        extended_states::Bool = false) where T<:AbstractFloat

    #Create physics data with one extra step ahead extra
    trange = collect(range(tspan[1], tspan[2], length = datasize))
    dt = trange[2]-trange[1]
    tsteps = push!(trange, dt + trange[end])
    tspan_new = (tspan[1], dt+tspan[2])
    physics_model_data = prior_model(u0, tspan_new, tsteps)
    physics_informed_data = vcat(train_data, physics_model_data[:, 1:end-1])

    in_size = size(physics_informed_data, 1)
    out_size = size(train_data, 1)
    res_size = size(W, 1)

    if size(W_in, 1) != res_size
        throw(DimensionMismatch("size(W_in, 1) must be equal to size(W, 1)"))
    elseif size(W_in, 2) != in_size
        throw(DimensionMismatch("size(W_in, 2) must be equal to in_size"))
    end

    states = states_matrix(W, W_in, physics_informed_data, alpha, activation, extended_states)

    return HESN{T, typeof(train_data),
        typeof(res_size),
        typeof(extended_states),
        typeof(activation),
        typeof(nla_type),
        typeof(prior_model),
        typeof(u0)}(res_size, in_size, out_size, train_data, physics_model_data,
        prior_model, u0, tspan, datasize, dt, alpha, nla_type, activation, W, W_in, states, extended_states)
end


"""
    HESNpredict(esn::AbstractLeakyESN, predict_len::Int, prior_data::AbstractArray{Float64}, model_size::Int, W_out::AbstractArray{Float64})

Return prediction the a starting after the training time using HESN model.
"""

function HESNpredict(hesn::AbstractHESN,
    predict_len::Int,
    W_out::AbstractArray{Float64})

    output = zeros(Float64, hesn.out_size, predict_len)
    x = hesn.states[:, end]
    predict_tsteps = [hesn.tspan[2]+hesn.dt]
    [append!(predict_tsteps, predict_tsteps[end]+hesn.dt) for i in 1:predict_len]
    tspan_new = (hesn.tspan[2]+hesn.dt, predict_tsteps[end])
    u0 = hesn.physics_model_data[:, end]
    physics_prediction_data = hesn.prior_model(u0, tspan_new, predict_tsteps)[:, 2:end]

    for i=1:predict_len
        x_new = nla(hesn.nla_type, x)
        x_new = vcat(x_new, physics_prediction_data[:, i]) #<-- append states of prior model at current tstep
        out = (W_out*x_new) #<-- prediction w/ reservoir state & prior model values given solved W_out
        output[:, i] = out
        out = vcat(out, physics_prediction_data[:, i]) #<-- append states of prior model for input of next prediction
        x = leaky_fixed_rnn(hesn.activation, hesn.alpha, hesn.W, hesn.W_in, x, out)
    end
    return output
end
