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
W_out = ESNtrain(Ridge(beta, Analytical()), esn)
@test size(W_out) == (out_size, esn.res_size)

#test train Lasso
W_out = ESNtrain(Lasso(beta, ProxGrad()), esn)
@test size(W_out) == (out_size, esn.res_size)

#test train ElastNet
W_out = ESNtrain(ElastNet(beta, beta, ProxGrad()), esn)
@test size(W_out) == (out_size, esn.res_size)

#test train RobustHuber
W_out = ESNtrain(RobustHuber(delta, beta, 0.0, Newton()), esn)
@test size(W_out) == (out_size, esn.res_size)
