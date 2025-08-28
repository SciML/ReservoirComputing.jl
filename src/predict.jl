@doc raw"""
    predict(rc, steps::Integer, ps, st; initialdata=nothing)
    predict(rc, data::AbstractMatrix, ps, st)

Run the model either in (1) closed-loop (auto-regressive) mode for a fixed number
of steps, or in (2) teacher-forced (point-by-point) mode over a given input
sequence.

## 1) Auto-regressive rollout

**Behavior**
- Rolls the model forward for `steps` time steps.
- At each step, the model’s output becomes the next input.

### Arguments
- `rc`: The reservoir chain / model.
- `steps::Integer`: Number of time steps to generate.
- `ps`: Model parameters (from `setup` or after `train!`).
- `st`: Model state (carry, RNG replicas, etc.), threaded across time.

### Keyword Arguments
- `initialdata=nothing`: Column vector used as the first input.

### Returns

- `Y`: Generated outputs of shape `(out_dims, steps)`.
- `st_out`: Final model state after `steps` steps.


## 2) Teacher-forced / point-by-point

- Feeds each column of `data` as input; the model state is threaded across time,
  and an output is produced for each input column.

### Arguments

- `rc`: The reservoir chain / model.
- `data`: Input sequence of shape `(in_dims, T)` (columns are time).
- `ps`: Model parameters.
- `st`: Initial model state before processing `data`.

### Returns

- `Y::AbstractMatrix`: Outputs for each input column, shape `(out_dims, T)`.
- `st_out`: Final model state after consuming all `T` columns.
"""
function predict(rc::AbstractLuxLayer,
    steps::Integer, ps, st; initialdata::AbstractVector)
    output = zeros(eltype(initialdata), length(initialdata), steps)
    for step in 1:steps
        initialdata, st = apply(rc, initialdata, ps, st)
        output[:, step] = initialdata
    end
    return output, st
end

function predict(rc::AbstractLuxLayer, data::AbstractMatrix, ps, st)
    T = size(data, 2)
    @assert T ≥ 1 "data must have at least one time step (columns)."

    y1, st = apply(rc, data[:, 1], ps, st)
    Y = similar(y1, size(y1, 1), T)
    Y[:, 1] .= y1

    @inbounds @views for t in 2:T
        yt, st = apply(rc, data[:, t], ps, st)
        Y[:, t] .= yt
    end
    return Y, st
end
