module ReservoirCellularAutomata

using CellularAutomata: CellularAutomaton
using Random: randperm
using Reexport: @reexport
using ReservoirComputing
@reexport using CellularAutomata
@reexport import ReservoirComputing: NLADefault, NLAT1, NLAT2, NLAT3, PartialSquare,
    ExtendedSquare,
    StandardStates, ExtendedStates, PaddedStates,
    PaddedExtendedStates, StandardRidge, Generative,
    Predictive, OutputLayer
import ReservoirComputing: train, AbstractReservoirComputer, AbstractOutputLayer,
    obtain_prediction, next_state_prediction!

include("reca.jl")
include("reca_input_encodings.jl")

export RECA
export RandomMapping, RandomMaps
export train

end
