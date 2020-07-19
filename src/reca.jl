abstract type AbstractReca <: AbstractEchoStateNetwork end

#cont values TO DO FURTHER TESTS
struct RECA{T<:AbstractFloat} <: AbstractReca
    res_size::Int
    in_size::Int
    out_size::Int
    rule::Int
    generations::Int
    train_data::AbstractArray{T}
    W_in::AbstractArray{T}
    nla_type::NonLinearAlgorithm
    states::AbstractArray{T}
end

function RECA(res_size::Int,
    train_data::AbstractArray{T},
    rule::Int,
    generations::Int,
    sigma::T; 
    nla_type = NLADefault()) where T<:AbstractFloat
    
    in_size = size(train_data, 1)
    out_size = size(train_data, 1)
    W_in = init_dense_input_layer(res_size, in_size, sigma)
    states = reca_states(train_data, W_in, res_size, rule, generations)
    
    return RECA{T}(res_size, in_size, out_size, rule, generations, train_data, W_in, nla_type, states)
end

function reca_states(input_data::AbstractArray{Float64},
    W_in::AbstractArray{Float64},
    res_size::Int, 
    rule::Int,
    generations::Int)
    
    input_size = size(input_data, 1)
    train_time = size(input_data, 2)
    
    states = Array{Int}(undef, res_size*generations, train_time)
    init_ca = zeros(Int, res_size)
    
    for i=1:train_time
        encoded_input = encode_data(input_data[:,i], W_in)
        init_ca = normal_xor(init_ca, encoded_input)
        ca = ECA(rule, init_ca, generations+1)
        ca_states = ca.cells[2:end ,:]
        states[:,i] = copy(reshape(transpose(ca_states), generations*res_size))
        init_ca = ca.cells[end, :]
    end
    return convert(AbstractArray{Float64}, states)
end 

function encode_data(input_vector::AbstractArray{Float64},
    W_in::AbstractArray{Float64})
    
    cont_values = W_in*input_vector
    norm_values = (cont_values.-minimum(cont_values))/(maximum(cont_values)-minimum(cont_values))
    return convert(AbstractArray{Int}, norm_values .> 0.5)
end

function normal_xor(old_x::AbstractArray{Int}, 
        next_x::AbstractArray{Int})

    new_x = old_x+next_x
    for i=1:size(new_x,1)
        if new_x[i]==0
            new_x[i]=0
        elseif new_x[i]==1
            new_x[i]=1
        elseif new_x[i]==2
            new_x[i]=0
        end
        
    end
    return new_x
end

function reca_predict(reca, predict_len, W_out)
    
    output = zeros(Float64, reca.in_size, predict_len)
    x = reca.states[:, end]
    init_ca = zeros(Int, reca.res_size)
    
    for i=1:predict_len
        x_new = nla(reca.nla_type, x)
        out = (W_out*x_new)
        output[:, i] = out
        encoded_input = encode_data(out, reca.W_in)
        init_ca = normal_xor(init_ca, encoded_input)
        ca = ECA(reca.rule, init_ca, reca.generations+1)
        ca_states = ca.cells[2:end ,:]
        x = convert(AbstractArray{Float64}, copy(reshape(transpose(ca_states), reca.generations*reca.res_size)))
        init_ca = ca.cells[end, :]
    end
    return output
end
