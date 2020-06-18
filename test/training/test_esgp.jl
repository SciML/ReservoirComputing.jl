using ReservoirComputing
using GaussianProcesses
#model parameters
approx_res_size = 30
radius = 1.2
activation = tanh
degree = 6
sigma = 0.1
beta = 0.0
alpha = 1.0
nla_type = NLADefault()
in_size = 1
out_size = 1
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

  
mean = MeanZero()
kernel = Lin(1.0)

gp = ESGPtrain(esn, mean, kernel, lognoise = -2.0, optimize = false)

esgp_output, sigma_esgp = ESGPpredict(esn, predict_len, gp)
@test isequal(size(esgp_output), size(test))
@test isequal(size(sigma_esgp), size(test))

esgp_output_h, sigma_esgp_h= ESGPpredict_h_steps(esn, predict_len, 1, test, gp)
@test isequal(size(esgp_output_h), size(test))
@test isequal(size(sigma_esgp_h), size(test))


