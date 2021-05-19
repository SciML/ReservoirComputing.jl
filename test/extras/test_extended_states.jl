using ReservoirComputing
using GaussianProcesses

#model parameters
const approx_res_size = 30
const radius = 1.2
const activation = tanh
const degree = 6
const sigma = 0.1
const beta = 0.1
const alpha = 1.0
const nla_type = NLADefault()
const in_size = 1
const out_size = 1
extended_states = true
const h_steps = 2


const train_len = 50
const predict_len = 12
data = ones(Float64, in_size, 100)
train = data[:, 1:1+train_len-1]
test = data[:, train_len:train_len+predict_len-1]

esn = ESN(approx_res_size,
    train,
    degree,
    radius,
    activation = activation,
    sigma = sigma,
    alpha = alpha,
    nla_type = nla_type,
    extended_states = extended_states)
            
W_out = ESNtrain(esn, beta)
@test size(W_out) == (out_size, esn.res_size+out_size)
#test predict
output = ESNpredict(esn, predict_len, W_out)
@test size(output) == (out_size, predict_len)
#test h steps predict
output = ESNpredict_h_steps(esn, predict_len, h_steps, test, W_out)
@test size(output) == (out_size, predict_len)

#test esnfitted
fit1 = ESNfitted(esn, W_out; autonomous=false)
@test size(fit1) == size(train)

fit2 = ESNfitted(esn, W_out; autonomous=true)
@test size(fit1) == size(train)

#test esgp
mean = MeanZero()
kernel = Lin(1.0)

gp = ESGPtrain(esn, mean, kernel, lognoise = -2.0, optimize = false)

esgp_output, sigma_esgp = ESGPpredict(esn, predict_len, gp)
@test size(output) == (out_size, predict_len)
esgp_output, sigma_esgp = ESGPpredict_h_steps(esn, predict_len, h_steps, test, gp)
@test size(output) == (out_size, predict_len)


#model parameters
const first_activation = tanh
const second_activation = identity
const first_lambda = 0.45
const second_lambda = 0.3


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
            
W_out = ESNtrain(esn, beta)
@test size(W_out) == (out_size, esn.res_size+out_size)
#test predict
output = dafESNpredict(esn, predict_len, W_out)
@test size(output) == (out_size, predict_len)
#test h steps predict
output = dafESNpredict_h_steps(esn, predict_len, h_steps, test, W_out)
@test size(output) == (out_size, predict_len)

