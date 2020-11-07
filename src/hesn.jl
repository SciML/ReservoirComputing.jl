abstract type AbstractHESN <: AbstractEchoStateNetwork end

struct HESN{T, S<:AbstractArray{T}, I, B, F, N, M} <: AbstractHESN
    res_size::I
    in_size::I
    out_size::I
    train_data::S
    prior_model::M
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

    physics_data = prior_model(u0, tspan, datasize)
    train_data = vcat(train_data, physics_data)
    in_size = size(train_data, 1)
    out_size = size(train_data, 1)
    res_size = size(W, 1)

    if size(W_in, 1) != res_size
        throw(DimensionMismatch("size(W_in, 1) must be equal to size(W, 1)"))
    elseif size(W_in, 2) != in_size
        throw(DimensionMismatch("size(W_in, 2) must be equal to in_size"))
    end

    states = states_matrix(W, W_in, train_data, alpha, activation, extended_states)

    return HESN{T, typeof(train_data),
        typeof(res_size),
        typeof(extended_states),
        typeof(activation),
        typeof(nla_type),
        typeof(prior_model)}(res_size, in_size, out_size, train_data, prior_model,
    alpha, nla_type, activation, W, W_in, states, extended_states)
end
