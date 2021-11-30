abstract type AbstractInputEncoding end
abstract type AbstractEncodingData end

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

struct RandomMaps{T,E,G,M,S} <: AbstractEncodingData
    permutations::T
    expansion_size::E
    generations::G
    maps::M
    states_size::S
end

function create_encoding(rm::RandomMapping, input_data, generations)
    maps = init_maps(size(input_data, 1), rm.permutations, rm.expansion_size)
    states_size = generations*rm.expansion_size*rm.permutations
    RandomMaps(rm.permutations, rm.expansion_size, generations, maps, states_size)
end



function reca_create_states(rm::RandomMaps, automata, input_data)
    
    train_time = size(input_data, 2)
    
    states = zeros(rm.states_size, train_time)
    init_ca = zeros(rm.expansion_size*rm.permutations)
    
    for i=1:train_time
        init_ca = encoding(rm, input_data[:,i], init_ca)
        ca = CellularAutomaton(automata, init_ca, rm.generations+1)
        ca_states = ca.evolution[2:end ,:]
        states[:,i] = reshape(transpose(ca_states), rm.generations*rm.expansion_size*rm.permutations)
        init_ca = ca.evolution[end, :]
    end
    states
end

function encoding(rm::RandomMaps, input_vector, tot_encoded_vector)
    
    input_size = size(input_vector,1)
    #single_encoded_size = Int(size(tot_encoded_vector, 1)/permutations)
    new_tot_enc_vec = copy(tot_encoded_vector)
        
    for i=1:rm.permutations
       new_tot_enc_vec[(i-1)*rm.expansion_size+1:i*rm.expansion_size]  = single_encoding(input_vector,
            new_tot_enc_vec[(i-1)*rm.expansion_size+1:i*rm.expansion_size], 
            rm.maps[i,:])
    end
    new_tot_enc_vec 
end

#function obtain_res_size(rm::RandomMapping, generations)
#    generations*rm.expansion_size*rm.permutations
#end
 
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
