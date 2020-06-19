using ReservoirComputing
using GaussianProcesses

#model parameters
approx_res_size = 30
radius = 1.2
activation = tanh
degree = 6
sigma = 0.1
beta = 0.1
alpha = 1.0
nla_type = NLADefault()
in_size = 1
out_size = 1
extended_states = true
h_steps = 2


train_len = 50
predict_len = 12
data = ones(Float64, in_size, 100)
train = data[:, 1:1+train_len-1]
test = data[:, train_len:train_len+predict_len-1]

esn = ESN(approx_res_size,
    train,
    degree,
    radius,
    activation,
    sigma,
    alpha,
    nla_type,
    extended_states)
            
W_out = ESNtrain(esn, beta)
@test size(W_out) == (out_size, esn.res_size+out_size)
#test predict
output = ESNpredict(esn, predict_len, W_out)
@test size(output) == (out_size, predict_len)
#test h steps predict
output = ESNpredict_h_steps(esn, predict_len, h_steps, test, W_out)
@test size(output) == (out_size, predict_len)

#test esgp
mean = MeanZero()
kernel = Lin(1.0)

gp = ESGPtrain(esn, mean, kernel, lognoise = -2.0, optimize = false)

esgp_output, sigma_esgp = ESGPpredict(esn, predict_len, gp)
@test size(output) == (out_size, predict_len)
esgp_output, sigma_esgp = ESGPpredict_h_steps(esn, predict_len, h_steps, test, gp)
@test size(output) == (out_size, predict_len)


#model parameters
first_activation = tanh
second_activation = identity
first_lambda = 0.45
second_lambda = 0.3


esn = dafESN(approx_res_size,
            train,
            degree,
            radius,
            first_lambda,
            second_lambda,
            first_activation,
            second_activation,
            sigma,
            alpha,
            nla_type,
            extended_states)
            
W_out = ESNtrain(esn, beta)
@test size(W_out) == (out_size, esn.res_size+out_size)
#test predict
output = dafESNpredict(esn, predict_len, W_out)
@test size(output) == (out_size, predict_len)
#test h steps predict
output = dafESNpredict_h_steps(esn, predict_len, h_steps, test, W_out)
@test size(output) == (out_size, predict_len)

