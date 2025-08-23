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

```
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

function train!(rc::ReservoirChain, train_data, target_data, ps, st, sr=StandardRidge(0.0);
    return_states::Bool=false)
    states, new_st = collectstates(rc, train_data, ps, st)
    W = train(sr, states, target_data)
    ps2, _ = addreadout!(rc, W, ps, new_st)
    return return_states ? ((ps2, new_st), states) : (ps2, new_st)
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

_quote_keys(t) = Expr(:tuple, (QuoteNode(s) for s in t)...)

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

    head_val = :((getfield(layers, 1) isa Readout)
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
