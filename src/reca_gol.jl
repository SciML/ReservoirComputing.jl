
function RECA_TwoDim(train_data, res_size, generations, permutations; nla_type = NLADefault())
    
    in_size = size(train_data, 1)
    out_size = size(train_data, 1)
    
    initial_state = zeros(Bool, res_size, res_size)
    maps = single_mapping(in_size, res_size, permutations)
    states = harvest_states_standard(train_data, initial_state, generations, permutations, maps)
    train_data = convert(AbstractArray{Float64}, train_data)
    
    return RECA_TwoDim(res_size, in_size, out_size, generations, permutations, train_data, nla_type, states, maps)
end

struct RECA_TwoDim <: AbstractReca
    res_size::Int
    in_size::Int
    out_size::Int
    generations::Int
    permutations::Int
    train_data::AbstractArray{Float64}
    nla_type::ReservoirComputing.NonLinearAlgorithm
    states::AbstractArray{Float64}
    maps::AbstractArray{Int}
end

function RECATD_predict_discrete(reca, 
    predict_len::Int, 
    W_out::AbstractArray{Float64})

    output = Array{Int}(undef, reca.in_size, predict_len)
    x = reca.states[:, end]
    last_states = reshape(x, reca.res_size, reca.res_size, reca.generations)
    last_state = convert(AbstractArray{Bool}, last_states[:,:, end])
    
    for i=1:predict_len
        out = convert(AbstractArray{Int}, W_out*x .> 0.5)
        output[:,i] = out#convert(AbstractArray{Int}, out .> 0.5)
        last_state = single_encoding(out, last_state, reca.maps)
        gol = GameOfLife(last_state, reca.generations)
        x = reshape(gol.all_runs, reca.res_size*reca.res_size*reca.generations)
        last_state = gol.all_runs[:, :, end]
    end
    return output
end

function harvest_states_standard(input_data::AbstractArray{Int}, 
    initial_state::AbstractArray{T},
    generations::Int,
    permutations::Int,
    single_map::AbstractArray{Int}) where T<: Bool
    
    input_size = size(input_data, 1)
    train_time = size(input_data, 2)
    res_size = size(initial_state, 1)
    
    states = zeros(T, res_size*res_size*generations , train_time)
    #single_map = single_mapping(input_size, res_size, permutations)#outside
    
    for i=1:train_time-1
        initial_state = single_encoding(input_data[:,i], initial_state, single_map)
        gol = GameOfLife(initial_state, generations)#+1
        gol_states = copy(gol.all_runs)#2:end
        states[:, i+1] = reshape(gol_states, res_size*res_size*generations)
        initial_state = gol.all_runs[:, :, end]
    end
    return states
end

function single_encoding(input_vector::AbstractArray{Int}, 
    reservoir::AbstractArray{T}, #must be size(reservoir, 1) = size(reservoir, 2)
    single_map::AbstractArray{Int}) where T<:Bool

    map_size = size(single_map, 2)
    input_size = size(input_vector, 1)
    res_size = size(reservoir, 1)
    permutations = Int(map_size/input_size)
    new_reservoir = copy(reservoir)
    
    counter = 0
    for i=1:permutations
        for j=1:input_size
            counter += 1
            first_coordinate = mod1(single_map[counter], res_size)
            second_coordinate = cld(single_map[counter], res_size)
            
            new_reservoir[first_coordinate, second_coordinate] = input_vector[j]
        end
    end
    
    return new_reservoir
end

#binary input onto reservoir first approach: random initial conditions, single mapping
function single_mapping(input_size::T, res_size::T, permutations::T) where T<: Int

    dims = input_size*permutations
    first_coordinate = sample(1:res_size*res_size, input_size*permutations, replace=false)
    
    single_map = collect(transpose(first_coordinate))
    
    return single_map
end
