using ReservoirComputing

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


train_len = 50
predict_len = 12
data = ones(Float64, in_size, 100)
train = data[:, 1:1+train_len-1]
test = data[:, train_len:train_len+predict_len-1] 

#test non linear algos
nla = [NLAT1(), NLAT2(), NLAT3()]
for t in nla
    nla_type = t
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
    @test size(W_out) == (out_size, esn.res_size)
    output = ESNpredict(esn, predict_len, W_out)
    @test size(output) == (out_size, predict_len)

end
