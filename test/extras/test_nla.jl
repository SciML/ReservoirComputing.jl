using ReservoirComputing

#model parameters
const approx_res_size = 30
const radius = 1.2
const activation = tanh
const degree = 6
const sigma = 0.1
const beta = 0.0
const alpha = 1.0
nla_type = NLADefault()
const in_size = 3
const out_size = 3
const extended_states = false


const train_len = 50
const predict_len = 12
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
        activation = activation,
        sigma = sigma,
        alpha = alpha,
        nla_type = nla_type,
        extended_states = extended_states)

    W_out = ESNtrain(esn, beta)
    @test size(W_out) == (out_size, esn.res_size)
    output = ESNpredict(esn, predict_len, W_out)
    @test size(output) == (out_size, predict_len)

end
