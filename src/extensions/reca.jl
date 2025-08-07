abstract type AbstractInputEncoding end
abstract type AbstractEncodingData end

"""
    RandomMapping(permutations, expansion_size)
    RandomMapping(permutations; expansion_size=40)
    RandomMapping(;permutations=8, expansion_size=40)

Random mapping of the input data directly in the reservoir. The `expansion_size`
determines the dimension of the single reservoir, and `permutations` determines the
number of total reservoirs that will be connected, each with a different mapping.
The detail of this implementation can be found in [1].

[1] Nichele, Stefano, and Andreas Molund. “Deep reservoir computing using cellular
automata.” arXiv preprint arXiv:1703.02806 (2017).
"""
struct RandomMapping{I, T} <: AbstractInputEncoding
    permutations::I
    expansion_size::T
end

struct RandomMaps{T, E, G, M, S} <: AbstractEncodingData
    permutations::T
    expansion_size::E
    generations::G
    maps::M
    states_size::S
    ca_size::S
end

abstract type AbstractReca <: AbstractReservoirComputer end

"""
    RECA(train_data,
        automata;
        generations = 8,
        input_encoding=RandomMapping(),
        nla_type = NLADefault(),
        states_type = StandardStates())

[1] Yilmaz, Ozgur. “_Reservoir computing using cellular automata._”
arXiv preprint arXiv:1410.0162 (2014).

[2] Nichele, Stefano, and Andreas Molund. “_Deep reservoir computing using cellular
automata._” arXiv preprint arXiv:1703.02806 (2017).
"""
struct RECA{S, R, E, T, Q} <: AbstractReca
    #res_size::I
    train_data::S
    automata::R
    input_encoding::E
    nla_type::ReservoirComputing.NonLinearAlgorithm
    states::T
    states_type::Q
end
