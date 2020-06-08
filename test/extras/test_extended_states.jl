using ReservoirComputing

#model parameters
approx_res_size = 30
radius = 1.2
activation = tanh
degree = 6
sigma = 0.1
beta = 0.1
alpha = 1.0
nla_type = NLADefault()
in_size = 3
out_size = 3
extended_states = true


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
