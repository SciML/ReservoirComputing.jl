using ReservoirComputing
using MLJLinearModels
#model parameters
approx_res_size = 30
radius = 1.2
activation = tanh
degree = 6
sigma = 0.1
beta = 0.0
alpha = 1.0
nla_type = NLADefault()
in_size = 3
out_size = 3
extended_states = false
delta = 0.5


train_len = 50
predict_len = 12
data = ones(Float64, in_size, 100)
train = data[:, 1:1+train_len-1]
test = data[:, train_len:train_len+predict_len-1]

#constructor 1
esn = ESN(approx_res_size,
    train,
    degree,
    radius,
    activation,
    sigma,
    alpha,
    nla_type,
    extended_states)

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
