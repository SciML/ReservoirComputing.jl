using ReservoirComputing

#model parameters
const res_size = 30
const radius = 1.2
const activation = tanh
const degree = 6
const sparsity = 0.5
const sigma = 0.1
const beta = 0.0
const alpha = 1.0
const nla_type = NLADefault()
const in_size = 3
const extended_states = false
const h_steps = 2
const max_value = 0.8

W_in = init_input_layer(res_size, in_size, sigma)

train_len = 50
predict_len = 12
data = ones(Float64, in_size, 100)
train = data[:, 1:1+train_len-1]
test = data[:, train_len:train_len+predict_len-1]

#test input layers functions
input_layer = [init_reservoir_givendeg, init_reservoir_givensp]

for t in input_layer

    if t == init_reservoir_givendeg
        W = t(res_size, radius, degree)
    else
        W = t(res_size, radius, sparsity)
    end
    esn = ESN(W,
        train,
        W_in,
        activation = activation,
        alpha = alpha,
        nla_type = nla_type,
        extended_states = extended_states) 

    @test size(W, 1) == res_size
end

W = pseudoSVD(res_size, max_value, sparsity)
@test size(W, 1) == res_size
