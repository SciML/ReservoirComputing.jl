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
#Let gamma be any number between 0 and 1
γ = rand()
model_in_size = 1

W = init_reservoir_givendeg(res_size, radius, degree)

train_len = 50
predict_len = 12
data = ones(Float64, in_size, 100)
train = data[:, 1:1+train_len-1]
test = data[:, train_len:train_len+predict_len-1]

#test input layers functions
input_layer = [init_input_layer, init_dense_input_layer, init_sparse_input_layer, min_complex_input, irrational_sign_input, physics_informed_input]

for t in input_layer

    if t == init_sparse_input_layer
        W_in = t(res_size, in_size, sigma, sparsity)

    elseif t == physics_informed_input
        W_in = t(res_size, in_size, sigma, γ, model_in_size)

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


#test physics informed input layer function
W_in = physics_informed_input(res_size, in_size, sigma, γ, model_in_size)

#Test num weights have been alotted correctly for state 1 according to the gamma chosen
@test sum(x->x!=0, W_in[:, 1]) == floor(Int, (res_size*γ)/(in_size - model_in_size))
#Test num weights have been alotted correctly for state 2 according to the gamma chosen
@test sum(x->x!=0, W_in[:, 2]) == floor(Int, (res_size*γ)/(in_size - model_in_size))
#Test num weights have been alotted correctly for model input 1 according to the gamma chosen
@test sum(x->x!=0, W_in[:, 3]) == floor(Int, (res_size*(1-γ))/(model_in_size))
#Test every reservoir node is connected exclusively to one state
@test sum(x->x=1, [sum(x->x!=0, W_in[i, :]) for i in 1:res_size]) == res_size
