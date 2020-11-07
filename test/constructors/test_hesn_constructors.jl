using ReservoirComputing

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
data = ones(Float64, in_size-out_size, 100)
train = data[:, 1:1+train_len-1]
#test = data[:, train_len:train_len+predict_len-1]


#user physics function
function lorenz()
end

#physics data generator for training and prediction
function prior_model(u0, tspan, datasize, model = lorenz)
    tsteps = range(tspan[1], tspan[2], length = datasize)
    sol = ones(length(u0), length(tsteps))
    return sol
end
concat_train = vcat(train, prior_model(u0, tspan, datasize))
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
@test isequal(concat_train, hesn.train_data)
@test isequal(alpha, hesn.alpha)
@test isequal(activation, hesn.activation)
@test isequal(nla_type, hesn.nla_type)
@test isequal(prior_model, hesn.prior_model)
@test size(hesn.W) == (hesn.res_size, hesn.res_size)
@test size(hesn.W_in) == (hesn.res_size, hesn.in_size)
@test size(hesn.states) == (hesn.res_size, train_len)
