using ReservoirComputing, DifferentialEquations, Statistics, Random

u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 1000.0)
datasize = 100000
tsteps = range(tspan[1], tspan[2]; length = datasize)

function lorenz(du, u, p, t)
    p = [10.0, 28.0, 8 / 3]
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

function prior_model_data_generator(u0, tspan, tsteps, model = lorenz)
    prob = ODEProblem(lorenz, u0, tspan)
    sol = Array(solve(prob; saveat = tsteps))
    return sol
end

train_len = 10000

ode_prob = ODEProblem(lorenz, u0, tspan)
ode_sol = solve(ode_prob; saveat = tsteps)
ode_data = Array(ode_sol)
input_data = ode_data[:, 1:train_len]
target_data = ode_data[:, 2:(train_len + 1)]

test_data = ode_data[:, (train_len + 1):end][:, 1:1000]
predict_len = size(test_data, 2)
tspan_train = (tspan[1], ode_sol.t[train_len])

km = KnowledgeModel(prior_model_data_generator, u0, tspan_train, train_len)

Random.seed!(77)
hesn = HybridESN(km,
    input_data,
    3,
    300;
    reservoir = rand_sparse)

output_layer = train(hesn, target_data, StandardRidge(0.3))

output = hesn(Generative(predict_len), output_layer)

@test mean(abs.(test_data[1:100] .- output[1:100])) ./ mean(abs.(test_data[1:100])) < 0.11
