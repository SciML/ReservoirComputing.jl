using ReservoirComputing  

#model parameters
shift = 1
approx_res_size = 30
N = 3
radius = 1.2
degree = 6
sigma = 0.1
in_size = N
out_size = N
train_len = 50
predict_len = 12
beta = 0.0
alpha = 1.0
nonlin_alg = "None"

data = ones(Float64, N, 100)
train = data[:, shift:shift+train_len-1]
test = data[:, train_len:train_len+predict_len-1]

#constructor
esn = ESN(approx_res_size,
    in_size,
    out_size,
    train,
    degree,
    sigma,
    alpha,
    beta,
    radius,
    nonlin_alg)

#test constructor
@test isequal(Integer(floor(approx_res_size/in_size)*in_size), esn.res_size)
@test isequal(in_size, esn.in_size)
@test isequal(out_size, esn.out_size)
@test isequal(train, esn.train_data)
@test isequal(degree, esn.degree)
@test isequal(sigma, esn.sigma)
@test isequal(alpha, esn.alpha)
@test isequal(beta, esn.beta)
@test isequal(radius, esn.radius)
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
      in_size,
      out_size,
      train,
      degree,
      sigma,
      alpha,
      beta,
      radius,
      nonlin_alg)
    W_out = ESNtrain(esn)
    @test size(W_out) == (out_size, esn.res_size)
    output = ESNpredict(esn, predict_len, W_out)
    @test size(output) == (out_size, predict_len)
    p_output = ESNsingle_predict(esn, predict_len, test[3,:], test, W_out)
    @test size(p_output) == (out_size, predict_len)
end
