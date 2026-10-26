module RCCellularAutomataExt
using ReservoirComputing: RECA, AbstractInputEncoding, ReservoirComputer,
    IntegerType, LinearReadout, StatefulLayer, __resolve_readout_in_dims
import ReservoirComputing: RECACell, RECA, RandomMapping, RandomMaps
using CellularAutomata: CellularAutomata, CellularAutomaton
using Random: Random, AbstractRNG, randperm

function create_encoding(
        rng::AbstractRNG, rm::RandomMapping, in_dims::IntegerType,
        generations::IntegerType
    )
    maps = init_maps(rng, in_dims, rm.permutations, rm.expansion_size)
    states_size = generations * rm.expansion_size * rm.permutations
    ca_size = rm.expansion_size * rm.permutations
    return RandomMaps(
        rm.permutations, rm.expansion_size, generations, maps, states_size, ca_size
    )
end

function encoding(rm::RandomMaps, input_vector, tot_encoded_vector)
    input_size = size(input_vector, 1)
    new_tot_enc_vec = copy(tot_encoded_vector)

    for i in 1:(rm.permutations)
        new_tot_enc_vec[((i - 1) * rm.expansion_size + 1):(i * rm.expansion_size)] = single_encoding(
            input_vector,
            new_tot_enc_vec[((i - 1) * rm.expansion_size + 1):(i * rm.expansion_size)],
            rm.maps[
                i,
                :,
            ]
        )
    end

    return new_tot_enc_vec
end

function single_encoding(input_vector, encoded_vector, map)
    length(map) == length(input_vector) || throw(
        DimensionMismatch(
            "RandomMaps mismatch: map length = $(length(map)) but input length = " *
                "$(length(input_vector)). (Build RandomMaps with in_dims = size(input, 1) " *
                "used at training time.)"
        )
    )
    new_enc_vec = copy(encoded_vector)

    for i in 1:size(input_vector, 1)
        new_enc_vec[map[i]] = input_vector[i]
    end

    return new_enc_vec
end

function init_maps(rng::AbstractRNG, input_size, permutations, mapped_vector_size)
    maps = Array{Int}(undef, permutations, input_size)

    for i in 1:permutations
        maps[i, :] = mapping(rng, input_size, mapped_vector_size)
    end

    return maps
end

function mapping(rng::AbstractRNG, input_size, mapped_vector_size)
    return randperm(rng, mapped_vector_size)[1:input_size]
end

function (reca::RECACell)((inp, (ca_prev,)), ps, st::NamedTuple)
    rm = reca.enc
    T = eltype(inp)
    ca0 = encoding(rm, inp, __ca_state_type(reca.automaton, T).(ca_prev))
    ca = CellularAutomaton(reca.automaton, ca0, rm.generations + 1)
    evo = ca.evolution
    feat2T = evo[2:end, :]
    feats = reshape(permutedims(feat2T), rm.states_size)
    ca_last = evo[end, :]
    return (T.(feats), (T.(ca_last),)), st
end

__ca_state_type(::CellularAutomata.CCA, ::Type{T}) where {T} = T
__ca_state_type(_, ::Type) = Int

function (reca::RECACell)(inp::AbstractVector, ps, st::NamedTuple)
    ca = st.ca
    return reca((inp, (ca,)), ps, st)
end

function RECA(
        in_dims::IntegerType,
        out_dims::IntegerType,
        automaton;
        input_encoding::AbstractInputEncoding = RandomMapping(),
        generations::Integer = 8,
        state_modifiers = (),
        readout_activation = identity,
        readout_in_dims = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    rm = create_encoding(rng, input_encoding, in_dims, generations)
    cell = RECACell(automaton, rm)

    mods = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
        Tuple(state_modifiers) : (state_modifiers,)

    ro_dims = __resolve_readout_in_dims(
        readout_in_dims, mods, rm.states_size, Int(in_dims)
    )
    ro = LinearReadout(ro_dims => out_dims, readout_activation)

    return ReservoirComputer(StatefulLayer(cell), mods, ro)
end

end #module
