@doc raw"""

    StandardRidge([Type], [reg])

Returns a training method for `train` based on ridge regression.
The equations for ridge regression are as follows:

```math
\mathbf{w} = (\mathbf{X}^\top \mathbf{X} +
\lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}
```

# Arguments
 - `Type`: type of the regularization argument. Default is inferred internally,
   there's usually no need to tweak this
 - `reg`: regularization coefficient. Default is set to 0.0 (linear regression).
"""
struct StandardRidge
    reg::Number
end

function StandardRidge(::Type{T}, reg) where {T<:Number}
    return StandardRidge(T.(reg))
end

function StandardRidge()
    return StandardRidge(0.0)
end


function _apply_washout(states::AbstractMatrix, targets::AbstractMatrix, washout::Integer)
    @assert washout ≥ 0 "washout must be ≥ 0"
    len_states = size(states, 2)
    @assert washout < len_states "washout=$washout is ≥ number of time steps=$len_states"
    first_idx = washout + 1
    states_wo = states[:, washout+1:end]
    targets_wo = targets[:, washout+1:end]
    return states_wo, targets_wo
end

@doc raw"""
    train!(rc::ReservoirChain, train_data, target_data, ps, st,
           sr::StandardRidge=StandardRidge(0.0);
           washout::Int=0, return_states::Bool=false)

Trains the Reservoir Computer by creating the reservoir states from `train_data`,
and then fiting the last [`LinearReadout`](@ref) layer by (ridge)
linear regression onto `target_data`. The learned weights are written into `ps`, and.
The returned state is the final state after running through the full sequence.

## Arguments

- `rc`: A [`ReservoirChain`](@ref) whose last trainable layer is a `LinearReadout`.
- `train_data`: input sequence (columns are time steps).
- `target_data`: targets aligned with `train_data`.
- `ps, st`: current parameters and state.
- `sr`: ridge spec, e.g. `StandardRidge(1e-4)`; `0.0` gives ordinary least squares.

## Keyword arguments

- `washout`: number of initial time steps to discard (applied equally to features
  and targets). Must satisfy `0 ≤ washout < T`. Default `0`.
- `return_states`: if `true`, also returns the feature matrix used
  for the fit.

## Returns

- `(ps2, st_after)` — updated parameters and the final model state.
- If `return_states=true`, also returns `states_used`.

## Notes

- Features are produced by `collectstates(rc, train_data, ps, st)`. If you rely on
  the implicit collection of a [`LinearReadout`](@ref), make sure that readout was created with
  `include_collect=true`, or insert an explicit [`Collect()`](@ref) earlier in the chain.
"""
function train!(rc::ReservoirChain, train_data, target_data, ps, st,
    train_method=StandardRidge(0.0);
    washout::Int=0, return_states::Bool=false)
    states, st_after = collectstates(rc, train_data, ps, st)
    states_wo, traindata_wo = washout > 0 ? _apply_washout(states, target_data, washout) : (states, target_data)
    output_matrix = train(train_method, states_wo, traindata_wo)
    ps2, st_after = addreadout!(rc, output_matrix, ps, st_after)
    return return_states ? ((ps2, st_after), states_wo) : (ps2, st_after)
end

function train(sr::StandardRidge, states::AbstractArray, target_data::AbstractArray)
    n_states = size(states, 1)
    A = [states'; sqrt(sr.reg) * I(n_states)]
    b = [target_data'; zeros(n_states, size(target_data, 1))]
    F = qr(A)
    Wt = F \ b
    output_layer = Matrix(Wt')
    return output_layer
end

#_quote_keys(t) = Expr(:tuple, (QuoteNode(s) for s in t)...)

@generated function _setweight_rt(p::NamedTuple{K}, W) where {K}
    keys = K
    Kq = _quote_keys(keys)
    idx = findfirst(==(Symbol(:weight)), keys)

    terms = Any[]
    for i in 1:length(keys)
        push!(terms, (idx === i) ? :(W) : :(getfield(p, $i)))
    end

    if idx === nothing
        newK = _quote_keys((keys..., :weight))
        return :(NamedTuple{$newK}(($(terms...), W)))
    else
        return :(NamedTuple{$Kq}(($(terms...),)))
    end
end

@generated function _addreadout(layers::NamedTuple{K}, ps::NamedTuple{K}, W) where {K}
    if length(K) == 0
        return :(NamedTuple())
    end
    tailK = Base.tail(K)
    Kq = _quote_keys(K)
    tailKq = _quote_keys(tailK)

    head_val = :((getfield(layers, 1) isa LinearReadout)
                 ? _setweight_rt(getfield(ps, 1), W)
                 : getfield(ps, 1))

    tail_call = :(_addreadout(NamedTuple{$tailKq}(Base.tail(layers)),
        NamedTuple{$tailKq}(Base.tail(ps)),
        W))

    return :(NamedTuple{$Kq}(($head_val, Base.values($tail_call)...)))
end

function addreadout!(rc::ReservoirChain,
    W::AbstractMatrix,
    ps::NamedTuple,
    st::NamedTuple)
    @assert propertynames(rc.layers) == propertynames(ps)
    new_ps = _addreadout(rc.layers, ps, W)
    return new_ps, st
end
