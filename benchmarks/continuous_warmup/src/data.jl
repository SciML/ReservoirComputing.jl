using Random
using SciMLBase
using OrdinaryDiffEqTsit5

const LORENZ_λ_MAX = 0.9056  # typical max Lyapunov for classical Lorenz-63

function lorenz!(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
    return nothing
end

"""
    make_lorenz(; dt, tspan, shift, train_len, predict_len, u0, p)

Returns named tuple with `input_data`, `target_data`, `test_data`,
`dt`, and full `data`.
"""
function make_lorenz(;
        dt::Float64 = 0.02,
        t_end::Float64 = 200.0,
        shift::Int = 300,
        train_len::Int = 5000,
        # Match the #456 Lorenz probe (predict ≈ 1250 samples ≈ 22.6 t_λ at dt=0.02).
        predict_len::Int = 1250,
        u0 = [1.0, 0.0, 0.0],
        p = [10.0, 28.0, 8 / 3],
    )
    data_prob = ODEProblem(lorenz!, u0, (0.0, t_end), p)
    data = Array(solve(data_prob, Tsit5(); saveat = dt))

    need = shift + train_len + predict_len
    size(data, 2) ≥ need || throw(
        ArgumentError(
            "Lorenz series too short: need $need samples, got $(size(data, 2)). " *
                "Increase t_end."
        )
    )

    input_data = data[:, shift:(shift + train_len - 1)]
    target_data = data[:, (shift + 1):(shift + train_len)]
    test_data = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]

    return (
        data = data,
        input_data = input_data,
        target_data = target_data,
        test_data = test_data,
        dt = dt,
        shift = shift,
        train_len = train_len,
        predict_len = predict_len,
        λ_max = LORENZ_λ_MAX,
    )
end

"""Shorter Lorenz split for --smoke runs."""
function make_lorenz_smoke()
    return make_lorenz(;
        t_end = 80.0,
        shift = 100,
        train_len = 800,
        predict_len = 200,
    )
end
