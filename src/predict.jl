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
- `steps`: Number of time steps to generate.
- `ps`: Model parameters.
- `st`: Model states.

### Keyword Arguments

- `initialdata=nothing`: Column vector used as the first input.
  Has to be provided.

### Returns

- `output`: Generated outputs of shape `(out_dims, steps)`.
- `st`: Final model state after `steps` steps.


## 2) Teacher-forced / point-by-point

- Feeds each column of `data` as input; the model state is threaded across time,
  and an output is produced for each input column.

### Arguments

- `rc`: The reservoir chain / model.
- `data`: Input sequence of shape `(in_dims, T)` (columns are time).
- `ps`: Model parameters.
- `st`: Model states.

### Returns

- `output`: Outputs for each input column, shape `(out_dims, T)`.
- `st`: Updated final model states.
"""
function predict(
        rc::AbstractLuxLayer,
        steps::Integer, ps, st; initialdata::AbstractVector
    )
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

    for t in 2:T
        yt, st = apply(rc, data[:, t], ps, st)
        Y[:, t] .= yt
    end
    return Y, st
end

# Two-level dispatch on the reservoir field, mirroring `collectstates` / `_collectstates`.
# Continuous reservoirs (`AbstractSciMLProblemReservoir`) plug in their own `_predict`
# methods from `RCODEReservoirExt`; everything else hits the fallbacks below, which
# replicate the discrete `predict(::AbstractLuxLayer, …)` bodies above.
#
# Not every `AbstractReservoirComputer` subtype carries a `:reservoir` field —
# `DeepESN`, for instance, owns a tuple of cells under `:cells`. For those
# subtypes we cannot extract a "reservoir layer" to dispatch on, so we pass
# `nothing` and let the `::Any` fallback take the discrete loop. (Concrete
# types like `DeepESN` already provide their own specialised `collectstates`,
# and `predict` itself only depends on `apply(rc, …)`, which works through
# their own `(rc::DeepESN)(…)` call.)

function predict(
        rc::AbstractReservoirComputer, steps::Integer, ps, st;
        initialdata::AbstractVector
    )
    res = hasfield(typeof(rc), :reservoir) ? rc.reservoir : nothing
    return _predict(res, rc, steps, ps, st; initialdata = initialdata)
end

function predict(rc::AbstractReservoirComputer, data::AbstractMatrix, ps, st)
    res = hasfield(typeof(rc), :reservoir) ? rc.reservoir : nothing
    return _predict(res, rc, data, ps, st)
end

function _predict(
        ::AbstractSciMLProblemReservoir,
        ::AbstractReservoirComputer, ::Integer, ::Any, ::Any;
        initialdata::AbstractVector
    )
    return error(
        "Autoregressive `predict(rc, steps, ps, st; initialdata)` for a " *
            "`SciMLProblemReservoir` requires the `RCODEReservoirExt` extension. " *
            "Load `SciMLBase` and `DataInterpolations` (plus an OrdinaryDiffEq " *
            "solver package — `OrdinaryDiffEqTsit5`, `OrdinaryDiffEq`, …) to enable it."
    )
end

function _predict(
        ::AbstractSciMLProblemReservoir,
        ::AbstractReservoirComputer, ::AbstractMatrix, ::Any, ::Any
    )
    return error(
        "Teacher-forced `predict(rc, data, ps, st)` for a " *
            "`SciMLProblemReservoir` requires the `RCODEReservoirExt` extension. " *
            "Load `SciMLBase` and `DataInterpolations` (plus an OrdinaryDiffEq " *
            "solver package — `OrdinaryDiffEqTsit5`, `OrdinaryDiffEq`, …) to enable it."
    )
end

function _predict(
        ::Any, rc::AbstractReservoirComputer, steps::Integer, ps, st;
        initialdata::AbstractVector
    )
    output = zeros(eltype(initialdata), length(initialdata), steps)
    for step in 1:steps
        initialdata, st = apply(rc, initialdata, ps, st)
        output[:, step] = initialdata
    end
    return output, st
end

function _predict(::Any, rc::AbstractReservoirComputer, data::AbstractMatrix, ps, st)
    n_samples = size(data, 2)
    n_samples ≥ 1 || throw(
        ArgumentError(
            "predict input data must have at least one column, got $n_samples."
        )
    )

    input_cols = eachcol(data)
    first_output, st = apply(rc, first(input_cols), ps, st)
    outputs = similar(first_output, size(first_output, 1), n_samples)
    outputs[:, 1] .= first_output

    for (idx, input_col) in Iterators.drop(enumerate(input_cols), 1)
        current_output, st = apply(rc, input_col, ps, st)
        outputs[:, idx] .= current_output
    end
    return outputs, st
end
