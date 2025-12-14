const BoolType = Union{StaticBool, Bool, Val{true}, Val{false}}
const InputType = Tuple{<:AbstractArray, Tuple{<:AbstractArray}}
const IntegerType = Union{Integer, StaticInteger}
const RCFields = (:cells, :states_modifiers, :readout)

abstract type AbstractReservoirComputer{Fields} <: AbstractLuxContainerLayer{Fields} end

### from Lux's extended ops

function safe_getproperty(x, ::Union{Val{v}, StaticSymbol{v}}) where {v}
    return v in Base.propertynames(x) ? Base.getproperty(x, v) : nothing
end
@generated function safe_getproperty(x::NamedTuple{names}, ::Union{
        Val{v}, StaticSymbol{v}}) where {
        names, v}
    return v in names ? :(x.$v) : :(nothing)
end

function dense_bias(generic_mat::AbstractMatrix,
        generic_vec::AbstractVecOrMat,
        bias::AbstractVector)
    return generic_mat * generic_vec .+ bias
end

function dense_bias(generic_mat::AbstractMatrix,
        generic_vec::AbstractVecOrMat, ::Nothing)
    return generic_mat * generic_vec
end
