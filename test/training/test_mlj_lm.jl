using ReservoirComputing
using MLJLinearModels
#model parameters
const approx_res_size = 30
const radius = 1.2
const activation = tanh
const degree = 6
const sigma = 0.1
const beta = 0.0
const alpha = 1.0
const nla_type = NLADefault()
const in_size = 3
const out_size = 3
const extended_states = false
const delta = 0.5


const train_len = 50
const predict_len = 12
data = ones(Float64, in_size, 100)
train = data[:, 1:1+train_len-1]
test = data[:, train_len:train_len+predict_len-1]

#constructor 1
esn = ESN(approx_res_size,
    train,
    degree,
    radius,
    activation = activation,
    sigma = sigma,
    alpha = alpha,
    nla_type = nla_type,
    extended_states = extended_states)

#test train Ridge
t = Ridge(beta, Analytical())
@test isequal(t.lambda, beta)
@test isequal(t.solver, Analytical())
W_out = ESNtrain(t, esn)
@test size(W_out) == (out_size, esn.res_size)

#test train Lasso
t = Lasso(beta, ProxGrad())
@test isequal(t.lambda, beta)
@test isequal(t.solver, ProxGrad())
W_out = ESNtrain(t, esn)
@test size(W_out) == (out_size, esn.res_size)

#test train ElastNet
t = ElastNet(beta, beta, ProxGrad())
@test isequal(t.lambda, beta)
@test isequal(t.gamma, beta)
@test isequal(t.solver, ProxGrad())
W_out = ESNtrain(t, esn)
@test size(W_out) == (out_size, esn.res_size)

#test train RobustHuber
t = RobustHuber(delta, beta, 0.0, Newton())
@test isequal(t.delta, delta)
@test isequal(t.lambda, beta)
@test isequal(t.gamma, 0.0)
@test isequal(t.solver, Newton())
W_out = ESNtrain(t, esn)
@test size(W_out) == (out_size, esn.res_size)
