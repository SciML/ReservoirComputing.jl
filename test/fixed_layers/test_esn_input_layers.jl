using ReservoirComputing

#model parameters
res_size = 30
radius = 1.2
activation = tanh
degree = 6
sparsity = 0.5
sigma = 0.1
beta = 0.0
alpha = 1.0
nla_type = NLADefault()
in_size = 3
extended_states = false
h_steps = 2

W = init_reservoir_givendeg(res_size, radius, degree)

train_len = 50
predict_len = 12
data = ones(Float64, in_size, 100)
train = data[:, 1:1+train_len-1]
test = data[:, train_len:train_len+predict_len-1]

#test input layers functions
input_layer = [init_input_layer, init_dense_input_layer, init_sparse_input_layer, min_complex_input]

for t in input_layer

    if t == init_sparse_input_layer
        W_in = t(res_size, in_size, sigma, sparsity)
    else
        W_in = t(res_size, in_size, sigma)
    end
    esn = ESN(W,
        train,
        W_in,
        activation = activation,
        alpha = alpha,
        nla_type = nla_type,
        extended_states = extended_states) 

    @test size(W_in, 1) == res_size
    @test size(W_in, 2) == in_size
end
