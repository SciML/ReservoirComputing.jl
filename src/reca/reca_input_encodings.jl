abstract type AbstractInputEncoding end

struct RandomMapping{I,T} <: AbstractInputEncoding
    permutations::I
    expansion_size::T
end

function RandomMapping(;permutations=8, expansion_size=40)
    RandomMapping(permutations, expansion_size)
end

function RandomMapping(permutations; expansion_size=40)
    RandomMapping(permutations, expansion_size)
end

function create_states(rm::RandomMapping, automata, generations, input_data)
    
    input_size = size(input_data, 1)
    train_time = size(input_data, 2)
    
    maps = init_maps(in_size, rm.permutations, rm.expansion_size)
    states = zeros(generations*rm.expansion_size*rm.permutations, train_time)
    init_ca = zeros(rm.expansion_size*rm.permutations)
    
    for i=1:train_time
        init_ca = encoding(rm, input_data[:,i], init_ca, maps)
        ca = CellularAutomaton(automata, init_ca, generations+1)
        ca_states = ca.cells[2:end ,:]
        states[:,i] = reshape(transpose(ca_states), generations*expansion_size*permutations)
        init_ca = ca.cells[end, :]
    end
    states
end

function encoding(rm::RandomMapping, input_vector, tot_encoded_vector, maps)
    
    input_size = size(input_vector,1)
    #single_encoded_size = Int(size(tot_encoded_vector, 1)/permutations)
    new_tot_enc_vec = copy(tot_encoded_vector)
        
    for i=1:rm.permutations
       new_tot_enc_vec[(i-1)*rm.expansion_size+1:i*rm.expansion_size]  = single_encoding(input_vector,
            new_tot_enc_vec[(i-1)*rm.expansion_size+1:i*rm.expansion_size], 
            maps[i,:])
    end
    new_tot_enc_vec 
end

function obtain_res_size(rm::RandomMapping, generations)
    generations*rm.expansion_size*rm.permutations
end
 
function single_encoding(input_vector, encoded_vector, map)
    
    new_enc_vec = copy(encoded_vector)
    
    for i=1:size(input_vector, 1)
        new_enc_vec[map[i]] = input_vector[i]
    end
    new_enc_vec
end
 
function init_maps(input_size, permutations, mapped_vector_size)
    
    maps = Array{Int}(undef, permutations, input_size)
    #tot_size = input_size*permutations
    
    for i=1:permutations
        maps[i,:] = mapping(input_size, mapped_vector_size)
    end
    maps
end
 
function mapping(input_size, mapped_vector_size)

    sample(1:mapped_vector_size, input_size, replace=false)
end    
