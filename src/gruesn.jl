 
abstract type AbstractGRUESN <: AbstractEchoStateNetwork end

struct GRUESN{T, S<:AbstractArray{T}, I, B, F, N, G} <: AbstractGRUESN
    res_size::I
    in_size::I
    out_size::I
    train_data::S
    alpha::T
    nla_type::N
    activation::F
    W::S
    W_in::S
    gates::G
    states::S
    extended_states::B
end

"""
    GRUESN(W::AbstractArray{T}, train_data::AbstractArray{T}, W_in::AbstractArray{T} [, gates_weight::T, activation::Any, alpha::T, nla_type::NonLinearAlgorithm, extended_states::Bool])
    
Return a Gated Recurrent Unit [1] ESN struct

[1] Cho, Kyunghyun, et al. “Learning phrase representations using RNN encoder-decoder for statistical machine translation.” arXiv preprint arXiv:1406.1078 (2014).
"""
function GRUESN(W::AbstractArray{T},
        train_data::AbstractArray{T},
        W_in::AbstractArray{T};
        gates_weight::T = 0.9, 
        activation::Any = tanh,
        alpha::T = 1.0,
        nla_type::NonLinearAlgorithm = NLADefault(),
        extended_states::Bool = false) where T<:AbstractFloat

    in_size = size(train_data, 1)
    out_size = size(train_data, 1)
    res_size = size(W, 1)

    if size(W_in, 1) != res_size
        throw(DimensionMismatch("size(W_in, 1) must be equal to size(W, 1)"))
    elseif size(W_in, 2) != in_size
        throw(DimensionMismatch("size(W_in, 2) must be equal to in_size"))
    end
    
    
    gates = GRUGates(res_size, in_size, gates_weight)
    states = gru_states(W, W_in, train_data, alpha, activation, extended_states, gates_weight, gates)

    return GRUESN{T, typeof(train_data), 
        typeof(res_size), 
        typeof(extended_states), 
        typeof(activation), 
        typeof(nla_type), 
        typeof(gates)}(res_size, in_size, out_size, train_data,
    alpha, nla_type, activation, W, W_in, gates, states, extended_states)
end


"""
    GRUESNpredict(esn::AbstractGRUESN, predict_len::Int, W_out::AbstractArray{Float64})

Return the prediction for a given lenght of the constructed GRUESN
"""
function GRUESNpredict(esn::AbstractGRUESN,
    predict_len::Int,
    W_out::AbstractArray{Float64})

    output = zeros(Float64, esn.in_size, predict_len)
    x = esn.states[:, end]

    if esn.extended_states == false
        for i=1:predict_len
            x_new = nla(esn.nla_type, x)
            out = (W_out*x_new)
            output[:, i] = out
            x = gru(esn.gates, esn.activation, esn.W, esn.W_in, x, out)
        end
    elseif esn.extended_states == true
        for i=1:predict_len
            x_new = nla(esn.nla_type, x)
            out = (W_out*x_new)
            output[:, i] = out
            x = vcat(gru(esn.gates, esn.activation, esn.W, esn.W_in, x[1:esn.res_size], out), out) 
        end
    end
    return output
end

abstract type AbstractGates end

struct GRUGates{T, S<:AbstractArray{T}} <: AbstractGates
    U_r::S
    W_r::S
    b_r::S
    U_z::S
    W_z::S
    b_z::S
end

function GRUGates(res_size, in_size, gates_weight)
    U_r = irrational_sign_input(res_size, in_size, gates_weight, start = res_size*in_size)
    W_r = irrational_sign_input(res_size, res_size, gates_weight, start = 2*res_size*in_size)
    b_r = irrational_sign_input(res_size, 1, gates_weight, start = res_size*(2*in_size+res_size))
    U_z = irrational_sign_input(res_size, in_size, gates_weight, start = res_size*(2*in_size+res_size+1))
    W_z = irrational_sign_input(res_size, res_size, gates_weight, start = res_size*(3*in_size+res_size+1))
    b_z = irrational_sign_input(res_size, 1, gates_weight, start = res_size*(2*in_size+2*res_size+1))
    
    return GRUGates{typeof(gates_weight), typeof(U_r)}(U_r, W_r, b_r, U_z, W_z, b_z)
end

function gru(gates, activation, W, W_in, x, y)
    reset_gate = gates.U_r*y + gates.W_r*x+gates.b_r
    reset_gate = 1 ./(1 .+exp.(-reset_gate))
    update = activation.(W_in*y+W*(reset_gate.*x))
    update_gate = gates.U_z*y + gates.W_z*x + gates.b_z
    update_gate = 1 ./(1 .+exp.(-update_gate))
    return update_gate.*update + (1 .- update_gate) .*x
end

function gru_states(W::AbstractArray{Float64},
        W_in::AbstractArray{Float64},
        train_data::AbstractArray{Float64},
        alpha::Float64,
        activation::Function,
        extended_states::Bool, 
        gates_weight::Float64, 
        gates::AbstractGates)

    train_len = size(train_data, 2)
    res_size = size(W, 1)
    in_size = size(train_data, 1)
    states = zeros(Float64, res_size, train_len)
    
    for i=1:train_len-1
        states[:, i+1] = gru(gates, activation, W, W_in, states[:, i], train_data[:, i])
    end

    if extended_states == true
        ext_states = vcat(states, hcat(zeros(Float64, in_size), train_data[:,1:end-1]))
        return ext_states
    else
        return states
    end
end
