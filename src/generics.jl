const BoolType = Union{StaticBool, Bool, Val{true}, Val{false}}
const InputType = Tuple{<:AbstractArray, Tuple{<:AbstractArray}}
const IntegerType = Union{Integer, StaticInt}
const RCFields = (:cells, :states_modifiers, :readout)

"""
    AbstractReservoirComputer{Fields} <: AbstractLuxContainerLayer{Fields}

Developer interface for a Lux-compatible reservoir-computing container.

## Type parameters

- `Fields`: the `Tuple` of container field names reported to Lux. Reservoir
  containers use `(:reservoir, :states_modifiers, :readout)`.

## Required fields

Subtypes using the generic ReservoirComputing implementation must provide:

- `reservoir`: a Lux-compatible layer that produces reservoir features.
- `states_modifiers`: a `Tuple` of Lux-compatible layers applied in order to
  those features. Use `()` when no modifiers are required.
- `readout`: a Lux-compatible layer that maps features to model outputs.

## Extension contract

Implement the fields above and the ordinary Lux layer call contract. The
generic `LuxCore.initialparameters`, `LuxCore.initialstates`, function-call,
and [`collectstates`](@ref) methods then compose the three components. Each
component must accept its matching parameter and state entry and return
`(output, updated_state)` through `LuxCore.apply`.

`states_modifiers` must preserve the feature layout expected by `readout`.
Parameter and state containers returned by the Lux generic functions have the
named fields `reservoir`, `states_modifiers`, and `readout` in that order.

## Example

```julia
struct MyReservoirComputer <: AbstractReservoirComputer{
        (:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end
```

Use [`ReservoirComputer`](@ref) unless a new model type needs its own
construction or presentation API.
"""
abstract type AbstractReservoirComputer{Fields} <: AbstractLuxContainerLayer{Fields} end

### from Lux's extended ops

function safe_getproperty(x, ::Union{Val{v}, StaticSymbol{v}}) where {v}
    return v in Base.propertynames(x) ? Base.getproperty(x, v) : nothing
end
@generated function safe_getproperty(
        x::NamedTuple{names}, ::Union{
            Val{v}, StaticSymbol{v},
        }
    ) where {
        names, v,
    }
    return v in names ? :(x.$v) : :(nothing)
end

_cell_out_dims(cell) = cell.out_dims

function dense_bias(
        generic_mat::AbstractMatrix,
        generic_vec::AbstractVecOrMat,
        bias::AbstractVector
    )
    return generic_mat * generic_vec .+ bias
end

function dense_bias(
        generic_mat::AbstractMatrix,
        generic_vec::AbstractVecOrMat, ::Nothing
    )
    return generic_mat * generic_vec
end
