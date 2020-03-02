using ReservoirComputing  

#model parameters
shift = 1
approx_res_size = 30
radius = 1.2
activation = tanh
degree = 6
sigma = 0.1
train_len = 50
predict_len = 12
beta = 0.0
alpha = 1.0
nonlin_alg = "None"
in_size = 3
out_size = 3


data = ones(Float64, in_size, 100)
train = data[:, shift:shift+train_len-1]
test = data[:, train_len:train_len+predict_len-1]

#constructor
esn = ESN(approx_res_size,
    train,
    degree,
    radius,
    activation,
    sigma,
    alpha,
    beta,
    nonlin_alg)

#test constructor
@test isequal(Integer(floor(approx_res_size/in_size)*in_size), esn.res_size)
@test isequal(train, esn.train_data)
@test isequal(degree, esn.degree)
@test isequal(sigma, esn.sigma)
@test isequal(alpha, esn.alpha)
@test isequal(beta, esn.beta)
@test isequal(radius, esn.radius)
@test isequal(activation, esn.activation)
@test isequal(nonlin_alg, esn.nonlin_alg)
@test size(esn.W) == (esn.res_size, esn.res_size)
@test size(esn.W_in) == (esn.res_size, esn.in_size)
@test size(esn.states) == (esn.res_size, train_len)

#test train
W_out = ESNtrain(esn)
@test size(W_out) == (out_size, esn.res_size)
#test predict
output = ESNpredict(esn, predict_len, W_out)
@test size(output) == (out_size, predict_len)

#test single predict
p_output = ESNsingle_predict(esn, predict_len, test[3,:], test, W_out)
@test size(p_output) == (out_size, predict_len)

#test non linear algos
nla = ["T1", "T2", "T3"]
for t in nla
    nonlin_alg = t
    esn = ESN(approx_res_size,
        train,
        degree,
        radius,
        activation,
        sigma,
        alpha,
        beta,
        nonlin_alg)
        
    W_out = ESNtrain(esn)
    @test size(W_out) == (out_size, esn.res_size)
    output = ESNpredict(esn, predict_len, W_out)
    @test size(output) == (out_size, predict_len)
    p_output = ESNsingle_predict(esn, predict_len, test[3,:], test, W_out)
    @test size(p_output) == (out_size, predict_len)
end
