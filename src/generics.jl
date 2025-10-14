const BoolType = Union{StaticBool,Bool,Val{true},Val{false}}
const InputType = Tuple{<:AbstractArray,Tuple{<:AbstractArray}}
const IntegerType = Union{Integer,StaticInteger}
const RCFields = (:cells, :states_modifiers, :readout)

abstract type AbstractReservoirComputer{Fields} <: AbstractLuxContainerLayer{Fields} end
