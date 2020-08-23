abstract type AbstractReca <: AbstractEchoStateNetwork end

struct RECA_discrete <: AbstractReca
    res_size::Int
    in_size::Int
    out_size::Int
    rule::Int
    generations::Int
    expansion_size::Int
    permutations::Int
    train_data::AbstractArray{Float64}
    nla_type::ReservoirComputing.NonLinearAlgorithm
    states::AbstractArray{Float64}
    maps::AbstractArray{Int}
end

"""
    RECA_discrete(train_data::AbstractArray{Int}, rule::Int, generations::Int, 
    expansion_size::Int, permutations::Int [, nla_type]) 

Create a RECA struct as described in [1] and [2].

[1] Yilmaz, Ozgur. “Reservoir computing using cellular automata.” arXiv preprint arXiv:1410.0162 (2014).
[2] Nichele, Stefano, and Andreas Molund. “Deep reservoir computing using cellular automata.” arXiv preprint arXiv:1703.02806 (2017).

"""
function RECA_discrete(train_data::AbstractArray{Int},
    rule::Int,
    generations::Int,
    expansion_size::Int,
    permutations::Int;
    nla_type = NLADefault()) 
    
    in_size = size(train_data, 1)
    out_size = size(train_data, 1)
    res_size = generations*expansion_size*permutations
    
    maps = init_maps(in_size, permutations, expansion_size)
    states = reca_states_discrete(train_data, expansion_size, permutations, generations, rule, maps)
    train_data = convert(AbstractArray{Float64}, train_data)
    
    return RECA_discrete(res_size, in_size, out_size, rule, generations, expansion_size, permutations, train_data, nla_type, states, maps)
end

"""
    RECAdirect_predict_discrete(reca::AbstractReca, W_out::AbstractArray{Float64}, test_data::AbstractArray{Int})

Given the input data return the corresponding predicted output, as described in [1].

[1] Yilmaz, Ozgur. “Reservoir computing using cellular automata.” arXiv preprint arXiv:1410.0162 (2014).
"""
function RECAdirect_predict_discrete(reca::AbstractReca, 
    W_out::AbstractArray{Float64}, 
    test_data::AbstractArray{Int})
    
    predict_len = size(test_data, 2)
    output = Array{Int}(undef, size(W_out, 1), predict_len)
    init_ca = zeros(Int, reca.expansion_size*reca.permutations)
    
    for i=1:predict_len
        init_ca = encoding(test_data[:,i], init_ca, reca.maps)
        ca = ECA(reca.rule, init_ca, reca.generations+1)
        ca_states = ca.cells[2:end ,:]
        x = copy(reshape(transpose(ca_states), reca.generations*reca.expansion_size*reca.permutations))
        out = W_out*x
        init_ca = ca.cells[end, :]
        output[:,i] = convert(AbstractArray{Int}, out .> 0.5)
    end
    return output
end
 
function reca_states_discrete(input_data::AbstractArray{Int},
    expansion_size::Int, 
    permutations::Int,
    generations::Int, 
    rule::Int,
    maps::AbstractArray{Int})
    
    input_size = size(input_data, 1)
    train_time = size(input_data, 2)
    
    states = zeros(Int, generations*expansion_size*permutations, train_time)
    init_ca = zeros(Int, expansion_size*permutations)
    
    for i=1:train_time
        init_ca = encoding(input_data[:,i], init_ca, maps)
        ca = ECA(rule, init_ca, generations+1)
        ca_states = ca.cells[2:end ,:]
        states[:,i] = reshape(transpose(ca_states), generations*expansion_size*permutations)
        init_ca = ca.cells[end, :]
    end
    return convert(AbstractArray{Float64}, states)
end
 
function encoding(input_vector::AbstractArray{Int},
    tot_encoded_vector::AbstractArray{Int},
    maps::AbstractArray{Int})
    
    permutations = size(maps, 1)
    input_size = size(input_vector,1)
    single_encoded_size = Int(size(tot_encoded_vector, 1)/permutations)
    
    new_tot_enc_vec = copy(tot_encoded_vector)
        
    for i=1:permutations
       new_tot_enc_vec[(i-1)*single_encoded_size+1:i*single_encoded_size]  = single_encoding(input_vector,
            new_tot_enc_vec[(i-1)*single_encoded_size+1:i*single_encoded_size], 
            maps[i,:])
    end
    return new_tot_enc_vec 
end
 
function single_encoding(input_vector::AbstractArray{Int}, 
    encoded_vector::AbstractArray{Int},
    map::AbstractArray{Int})
    
    new_enc_vec = copy(encoded_vector)
    
    for i=1:size(input_vector, 1)
        new_enc_vec[map[i]] = input_vector[i]
    end
    return new_enc_vec
end
 
function init_maps(input_size::Int, 
        permutations::Int,
        mapped_vector_size::Int)
    
    maps = Array{Int}(undef, permutations, input_size)
    #tot_size = input_size*permutations
    
    for i=1:permutations
        maps[i,:] = mapping(input_size, mapped_vector_size)
    end
    return maps
end
 
function mapping(input_size::Int, mapped_vector_size::Int)

    map = sample(1:mapped_vector_size, input_size, replace=false)
    return map
end    
