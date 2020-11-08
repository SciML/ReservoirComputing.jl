using ReservoirComputing, MLJLinearModels


#model parameters
const approx_res_size = 30
const radius = 1.2
const activation = tanh
const degree = 6
const sigma = 0.1
const beta = 0.0
const alpha = 1.0
const nla_type = NLADefault()
const in_size = 6
const out_size = 3
const γ = rand()
const prior_model_size = 3
const u0 = [1.0,0.0,0.0]
const tspan = (0.0,1000.0)
const datasize = 50
W_in = ReservoirComputing.physics_informed_input(approx_res_size, in_size, sigma, γ, prior_model_size)
W = ReservoirComputing.init_reservoir_givendeg(approx_res_size, radius, degree)
const extended_states = false
const h_steps = 2


const train_len = 50
const predict_len = 12
data = ones(Float64, in_size-out_size, 100).+γ
train = data[:, 1:train_len]


#user physics function
function lorenz()
end

#physics data generator for training and prediction
trange = collect(range(tspan[1], tspan[2], length = train_len))
dt = trange[2]-trange[1]
tsteps = push!(trange, dt + trange[end])
tspan_new = (tspan[1], dt+tspan[2])

function prior_model(u0, tspan_new, tsteps, model = lorenz)
    sol = ones(length(u0), length(tsteps)).*γ
    return sol
end
physics_model_data = prior_model(u0, tspan_new, tsteps)

#constructor 1
hesn = HESN(W,
    train,
    prior_model,
    u0,
    tspan,
    datasize,
    W_in,
    activation = activation,
    alpha = alpha,
    nla_type = nla_type,
    extended_states = extended_states)


#test constructor
@test isequal(Integer(floor(approx_res_size/in_size)*in_size), hesn.res_size)
@test isequal(train, hesn.train_data)
@test isequal(prior_model(u0, tspan_new, tsteps), hesn.physics_model_data)
@test isequal(vcat(train, physics_model_data[:, 1:end-1]), vcat(hesn.train_data, hesn.physics_model_data[:,1:end-1]))
@test isequal(alpha, hesn.alpha)
@test isequal(activation, hesn.activation)
@test isequal(nla_type, hesn.nla_type)
@test isequal(prior_model, hesn.prior_model)
@test isequal(datasize, hesn.datasize)
@test isequal(u0, hesn.u0)
@test isequal(tspan, hesn.tspan)
@test isequal(dt, hesn.dt)
@test size(hesn.W) == (hesn.res_size, hesn.res_size)
@test size(hesn.W_in) == (hesn.res_size, hesn.in_size)
@test size(hesn.states) == (hesn.res_size, train_len)


<<<<<<< HESN-Constructor

#test dimension mismatch of hesn constructor
bad_in_size = 4
W_in = ReservoirComputing.physics_informed_input(approx_res_size, bad_in_size, sigma, γ, prior_model_size)
@test_throws DimensionMismatch hesn2 = HESN(W, train, prior_model, u0, tspan, datasize, W_in, activation = activation, alpha = alpha, nla_type = nla_type, extended_states = extended_states)

bad_res_size = approx_res_size-1
W_in = ReservoirComputing.physics_informed_input(bad_res_size, in_size, sigma, γ, prior_model_size)
@test_throws DimensionMismatch hesn2 = HESN(W, train, prior_model, u0, tspan, datasize, W_in, activation = activation, alpha = alpha, nla_type = nla_type, extended_states = extended_states)
=======
#test train
linear_model = Ridge(beta, Analytical())
W_out = HESNtrain(linear_model, hesn)
@test size(W_out) == (out_size, hesn.res_size+size(hesn.physics_model_data,1))
>>>>>>> init commit for HESN training + tests
