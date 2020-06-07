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
        activation,
        alpha,
        nla_type,
        extended_states) 

    @test size(W, 1) == res_size
end
 
