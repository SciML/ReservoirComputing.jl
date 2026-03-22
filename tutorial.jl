using OrdinaryDiffEq
using Plots
using Random
using LinearAlgebra
using DataInterpolations
using ReservoirComputing

Random.seed!(42)
rng = MersenneTwister(17)

function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

prob = ODEProblem(lorenz, [1.0f0, 0.0f0, 0.0f0],
                  (0.0, 200.0),
                  Float32[10.0, 28.0, 8/3])

data = Array(solve(prob, ABM54(); dt=0.02f0))


shift = 300
train_len = 5000
predict_len = 1250

input_data  = data[:, shift:(shift + train_len - 1)]
target_data = data[:, (shift + 1):(shift + train_len)]
test = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]

ctesn = CTESN(3, 300, 3;
    init_reservoir = rand_sparse(; radius=1.2, sparsity=6/300),
    state_modifiers = NLAT2
)

ps, st = setup(rng, ctesn)

ps, st = train!(ctesn, input_data, target_data, ps, st)

# forcing predict
# y_func, st = predict(ctesn, test, ps, st)

# Autoregressive predict
y_func, _ = predict(ctesn, predict_len, ps, st; initialdata=test[:, 1])

ts = collect(0:predict_len-1)
pred = hcat([y_func(t) for t in ts]...)

plot(transpose(pred)[:, 1],
     transpose(pred)[:, 2],
     transpose(pred)[:, 3],
     label="predicted",
     size=(1200, 900),
     linewidth=3)

plot!(transpose(test)[:, 1],
      transpose(test)[:, 2],
      transpose(test)[:, 3],
      label="actual",
      linewidth=3,
      alpha=0.8)