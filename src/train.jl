@doc raw"""

    StandardRidge([Type], [reg])

Ridge regression method.

## Equations

```math
\mathbf{w} = (\mathbf{X}^\top \mathbf{X} +
\lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}
```

## Arguments

 - `Type`: type of the regularization argument. Default is inferred internally,
   there's usually no need to tweak this
 - `reg`: regularization coefficient. Default is set to 0.0 (linear regression).
"""
struct StandardRidge
    reg::Number
end

function StandardRidge(::Type{T}, reg) where {T <: Number}
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
    states_wo = states[:, (washout + 1):end]
    targets_wo = targets[:, (washout + 1):end]
    return states_wo, targets_wo
end

_set_readout(ps, m::ReservoirChain, W) = first(addreadout!(m, W, ps, NamedTuple()))

abstract type AbstractReservoirComputingSolver end

struct QRSolver <: AbstractReservoirComputingSolver end

@doc raw"""
    train(train_method, states, target_data; kwargs...)

Lower level training hook to fit a readout from precomputed
reservoir features and given targets.

Dispatching on this method with different training methods
allows one to hook directly into [`train!`](@ref) without
additional changes.

## Arguments

- `train_method`: An object describing the training algorithm and its hyperparameters
  (e.g. regularization strength, solver choice, constraints).
- `states`: Feature matrix with reservoir states (ie. obtained with [`collectstates`](@ref)).
  Shape `(n_features, T)`, where `T` is the number of samples (e.g. time steps).
- `target_data`: Target matrix aligned with `states`. Shape `(n_outputs, T)`.

## Returns

- `output_weights`: Trained readout. Should be a forward method to be hooked into a
  layer. For instance, in case of linear regression `output_weights` is a mtrix
  consumable by [`LinearReadout`](@ref).

## Notes

- Any sequence pre-processing (e.g. washout) should be handled by the caller before
  invoking `train`. See [`train!`](@ref) for an end-to-end workflow.
- For very long `T`, consider chunked or iterative solvers to reduce memory usage.
- If your approach returns additional artifacts (e.g. diagnostics), prefer storing
  them inside `train_method` or exposing a separate API; keep `train`’s return
  value as the forward method only.
"""
function train(
        sr::StandardRidge, states::AbstractMatrix, target_data::AbstractMatrix;
        solver = QRSolver(), kwargs...
    )
    return _train_ridge(solver, sr, states, target_data; kwargs...)
end

function _train_ridge(::QRSolver, sr::StandardRidge,
        states::AbstractMatrix, target_data::AbstractMatrix; kwargs...)
    n_states = size(states, 1)
    A = [states'; sqrt(sr.reg) * I(n_states)]
    b = [target_data'; zeros(eltype(target_data), n_states, size(target_data, 1))]
    F = qr(A)
    Wt = F \ b
    output_layer = Matrix(Wt')
    return output_layer
end

@doc raw"""
    train!(rc, train_data, target_data, ps, st,
           train_method=StandardRidge(0.0);
           washout=0, return_states=false)

Trains a given reservoir computing by creating the reservoir states from `train_data`,
and then fiting the readout layer using `target_data` as target.
The learned weights/layer are written into `ps`, while the reservoir states are written
in `st`.

## Arguments

- `rc`: A reservoir computing model, either provided by ReservoirComputing.jl
  or built with [`ReservoirChain`](@ref). Must contain a trainable layer
  (for example [`LinearReadout`](@ref)), and a collection point [`Collect`](@ref).
- `train_data`: input sequence where columns are time steps.
- `target_data`: targets aligned with `train_data`.
- `ps`: model parameters.
- `st`: model states.
- `train_method`: training algorithm. Default is [`StandardRidge`](@ref).

## Keyword arguments

- `washout`: number of initial time steps to discard (applied equally to features
  and targets). Default `0`.
- `return_states`: if `true`, also returns the feature matrix used
  for the fit.
- `kwargs...`: additional keyword arguments for the training algorithm, if needed.
  Defaults vary according to the different training method.

## Returns

- `(ps, st)`: updated model parameters and states.
- `(ps, st), states`: If `return_states=true`.

## Notes

- Features are produced by `collectstates(rc, train_data, ps, st)`. If you rely on
  the implicit collection of a [`LinearReadout`](@ref), make sure that readout was created with
  `include_collect=true`, or insert an explicit [`Collect()`](@ref) earlier in the
  [`ReservoirChain`](@ref).
"""
function train!(
        rc, train_data, target_data, ps, st,
        train_method = StandardRidge(0.0);
        washout::Int = 0, return_states::Bool = false, kwargs...
    )
    raw_states, st_after = collectstates(rc, train_data, ps, st)
    states_wo,
        traindata_wo = washout > 0 ? _apply_washout(raw_states, target_data, washout) :
        (raw_states, target_data)
    output_matrix = train(train_method, states_wo, traindata_wo; kwargs...)
    ps2, st_after = addreadout!(rc, output_matrix, ps, st_after)
    st_after = merge(st_after, (; :states => states_wo))
    return return_states ? ((ps2, st_after), states_wo) : (ps2, st_after)
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

    head_val = :(
        (getfield(layers, 1) isa LinearReadout)
            ? _setweight_rt(getfield(ps, 1), W)
            : getfield(ps, 1)
    )

    tail_call = :(
        _addreadout(
            NamedTuple{$tailKq}(Base.tail(layers)),
            NamedTuple{$tailKq}(Base.tail(ps)),
            W
        )
    )

    return :(NamedTuple{$Kq}(($head_val, Base.values($tail_call)...)))
end

function addreadout!(
        rc::ReservoirChain,
        W::AbstractMatrix,
        ps::NamedTuple,
        st::NamedTuple
    )
    @assert propertynames(rc.layers) == propertynames(ps)
    new_ps = _addreadout(rc.layers, ps, W)
    return new_ps, st
end
