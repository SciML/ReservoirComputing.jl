module RCCellularAutomataExt
using ReservoirComputing: RECA, AbstractInputEncoding, ReservoirComputer,
                          IntegerType, LinearReadout, StatefulLayer
import ReservoirComputing: RECACell, RECA, RandomMapping, RandomMaps
using CellularAutomata
using Random: randperm

function create_encoding(rm::RandomMapping, in_dims::IntegerType, generations::IntegerType)
    maps = init_maps(in_dims, rm.permutations, rm.expansion_size)
    states_size = generations * rm.expansion_size * rm.permutations
    ca_size = rm.expansion_size * rm.permutations
    return RandomMaps(
        rm.permutations, rm.expansion_size, generations, maps, states_size, ca_size)
end

function encoding(rm::RandomMaps, input_vector, tot_encoded_vector)
    input_size = size(input_vector, 1)
    #single_encoded_size = Int(size(tot_encoded_vector, 1)/permutations)
    new_tot_enc_vec = copy(tot_encoded_vector)

    for i in 1:(rm.permutations)
        new_tot_enc_vec[((i - 1) * rm.expansion_size + 1):(i * rm.expansion_size)] = single_encoding(
            input_vector,
            new_tot_enc_vec[((i - 1) * rm.expansion_size + 1):(i * rm.expansion_size)],
            rm.maps[i,
            :])
    end

    return new_tot_enc_vec
end

function single_encoding(input_vector, encoded_vector, map)
    @assert length(map)==length(input_vector) """
      RandomMaps mismatch: map length = $(length(map)) but input length = $(length(input_vector)).
      (Build RandomMaps with in_dims = size(input, 1) used at training time.)
      """
    new_enc_vec = copy(encoded_vector)

    for i in 1:size(input_vector, 1)
        new_enc_vec[map[i]] = input_vector[i]
    end

    return new_enc_vec
end

function init_maps(input_size, permutations, mapped_vector_size)
    maps = Array{Int}(undef, permutations, input_size)
    #tot_size = input_size*permutations

    for i in 1:permutations
        maps[i, :] = mapping(input_size, mapped_vector_size)
    end

    return maps
end

function mapping(input_size, mapped_vector_size)
    #sample(1:mapped_vector_size, input_size; replace=false)
    return randperm(mapped_vector_size)[1:input_size]
end

function (reca::RECACell)((inp, (ca_prev,)), ps, st::NamedTuple)
    rm = reca.enc
    T = eltype(inp)
    ca0 = T.(encoding(rm, inp, T.(ca_prev)))
    ca = CellularAutomaton(reca.automaton, ca0, rm.generations + 1)
    evo = ca.evolution
    feat2T = evo[2:end, :]
    feats = reshape(permutedims(feat2T), rm.states_size)
    ca_last = evo[end, :]
    return (T.(feats), (T.(ca_last),)), st
end

function (reca::RECACell)(inp::AbstractVector, ps, st::NamedTuple)
    ca = st.ca
    return reca((inp, (ca,)), ps, st)
end

function RECA(in_dims::IntegerType,
        out_dims::IntegerType,
        automaton;
        input_encoding::AbstractInputEncoding = RandomMapping(),
        generations::Integer = 8,
        state_modifiers = (),
        readout_activation = identity)
    rm = create_encoding(input_encoding, in_dims, generations)
    cell = RECACell(automaton, rm)

    mods = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
           Tuple(state_modifiers) : (state_modifiers,)

    ro = LinearReadout(rm.states_size => out_dims, readout_activation)

    return ReservoirComputer(StatefulLayer(cell), mods, ro)
end

end #module
