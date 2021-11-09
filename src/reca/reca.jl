abstract type AbstractReca <: AbstractReservoirComputer end

struct RECA{I,S,R,E,T} <: AbstractReca
    res_size::I
    train_data::S
    automata::R
    input_encoding::E
    nla_type::ReservoirComputing.NonLinearAlgorithm
    states::T
end

"""
    RECA()

[1] Yilmaz, Ozgur. “Reservoir computing using cellular automata.” arXiv preprint arXiv:1410.0162 (2014).
[2] Nichele, Stefano, and Andreas Molund. “Deep reservoir computing using cellular automata.” arXiv preprint arXiv:1703.02806 (2017).
"""
function RECA(train_data,
    automata;
    input_encoding=RandomMapping(),
    nla_type = NLADefault()) 
    
    in_size = size(train_data, 1)
    res_size = obtain_res_size(input_encoding, generations)
    states = create_states(input_encoding, automata, generations, input_data)
    
    RECA(res_size, train_data, automata, input_encoding, nla_type, states)
end


