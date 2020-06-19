using ReservoirComputing

#model parameters
const approx_res_size = 30
const radius = 1.2
const degree = 6
const first_activation = tanh
const second_activation = identity
const first_lambda = 0.45
const second_lambda = 0.3
const sigma = 0.3
const beta = 0.0
const alpha = 1.0
const nla_type = NLADefault()
const in_size = 3
const out_size = 3
const extended_states = false
const h_steps = 2

W_in = init_input_layer(approx_res_size, in_size, sigma)
W = init_reservoir_givendeg(approx_res_size, radius, degree)

const train_len = 50
const predict_len = 12
data = ones(Float64, in_size, 100)
train = data[:, 1:1+train_len-1]
test = data[:, train_len:train_len+predict_len-1]

#constructor 1
esn = dafESN(approx_res_size,
            train,
            degree,
            radius,
            first_lambda,
            second_lambda,
            first_activation = first_activation,
            second_activation = second_activation,
            sigma = sigma,
            alpha = alpha,
            nla_type = nla_type,
            extended_states = extended_states)

#test constructor
@test isequal(Integer(floor(approx_res_size/in_size)*in_size), esn.res_size)
@test isequal(train, esn.train_data)
@test isequal(alpha, esn.alpha)
@test isequal(first_activation, esn.first_activation)
@test isequal(second_activation, esn.second_activation)
@test isequal(first_lambda, esn.first_lambda)
@test isequal(second_lambda, esn.second_lambda)
@test isequal(nla_type, esn.nla_type)
@test size(esn.W) == (esn.res_size, esn.res_size)
@test size(esn.W_in) == (esn.res_size, esn.in_size)
@test size(esn.states) == (esn.res_size, train_len)

#test train
W_out = ESNtrain(esn, beta)
@test size(W_out) == (out_size, esn.res_size)
#test predict
output = dafESNpredict(esn, predict_len, W_out)
@test size(output) == (out_size, predict_len)

#constructor 2
esn = dafESN(W,
            train,
            first_lambda,
            second_lambda,
            first_activation = first_activation,
            second_activation = second_activation,
            sigma = sigma,
            alpha = alpha,
            nla_type = nla_type,
            extended_states = extended_states)

#test constructor
@test isequal(Integer(floor(approx_res_size/in_size)*in_size), esn.res_size)
@test isequal(train, esn.train_data)
@test isequal(alpha, esn.alpha)
@test isequal(first_activation, esn.first_activation)
@test isequal(second_activation, esn.second_activation)
@test isequal(first_lambda, esn.first_lambda)
@test isequal(second_lambda, esn.second_lambda)
@test isequal(nla_type, esn.nla_type)
@test size(esn.W) == (esn.res_size, esn.res_size)
@test size(esn.W_in) == (esn.res_size, esn.in_size)
@test size(esn.states) == (esn.res_size, train_len)

#test train
W_out = ESNtrain(esn, beta)
@test size(W_out) == (out_size, esn.res_size)
#test predict
output = dafESNpredict(esn, predict_len, W_out)
@test size(output) == (out_size, predict_len)

#constructor 3
esn = dafESN(approx_res_size,
            train,
            degree,
            radius,
            first_lambda,
            second_lambda,
            W_in,
            first_activation = first_activation,
            second_activation = second_activation,
            alpha = alpha,
            nla_type = nla_type,
            extended_states = extended_states)

#test constructor
@test isequal(Integer(floor(approx_res_size/in_size)*in_size), esn.res_size)
@test isequal(train, esn.train_data)
@test isequal(alpha, esn.alpha)
@test isequal(first_activation, esn.first_activation)
@test isequal(second_activation, esn.second_activation)
@test isequal(first_lambda, esn.first_lambda)
@test isequal(second_lambda, esn.second_lambda)
@test isequal(nla_type, esn.nla_type)
@test size(esn.W) == (esn.res_size, esn.res_size)
@test size(esn.W_in) == (esn.res_size, esn.in_size)
@test size(esn.states) == (esn.res_size, train_len)

#test train
W_out = ESNtrain(esn, beta)
@test size(W_out) == (out_size, esn.res_size)
#test predict
output = dafESNpredict(esn, predict_len, W_out)
@test size(output) == (out_size, predict_len)

#constructor 4
esn = dafESN(W,
            train,
            first_lambda,
            second_lambda,
            W_in,
            first_activation = first_activation,
            second_activation = second_activation,
            alpha = alpha,
            nla_type = nla_type,
            extended_states = extended_states)

#test constructor
@test isequal(Integer(floor(approx_res_size/in_size)*in_size), esn.res_size)
@test isequal(train, esn.train_data)
@test isequal(alpha, esn.alpha)
@test isequal(first_activation, esn.first_activation)
@test isequal(second_activation, esn.second_activation)
@test isequal(first_lambda, esn.first_lambda)
@test isequal(second_lambda, esn.second_lambda)
@test isequal(nla_type, esn.nla_type)
@test size(esn.W) == (esn.res_size, esn.res_size)
@test size(esn.W_in) == (esn.res_size, esn.in_size)
@test size(esn.states) == (esn.res_size, train_len)

#test train
W_out = ESNtrain(esn, beta)
@test size(W_out) == (out_size, esn.res_size)
#test predict
output = dafESNpredict(esn, predict_len, W_out)
@test size(output) == (out_size, predict_len)

output = dafESNpredict_h_steps(esn, predict_len, h_steps, test, W_out)
@test size(output) == (out_size, predict_len)
