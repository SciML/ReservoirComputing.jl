using ReservoirComputing  

#model parameters
approx_res_size = 30
radius = 1.2
degree = 6
first_activation = tanh
second_activation = identity
first_lambda = 0.45
second_lambda = 0.3
sigma = 0.3
beta = 0.0
alpha = 1.0
nonlin_alg = NonLinAlgDefault
in_size = 3
out_size = 3

train_len = 50
predict_len = 12
data = ones(Float64, in_size, 100)
train = data[:, 1:1+train_len-1]
test = data[:, train_len:train_len+predict_len-1]

#constructor
esn = dafESN(approx_res_size,
            train,
            degree,
            radius,
            first_lambda,
            second_lambda,
            first_activation,
            second_activation,
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
@test isequal(first_activation, esn.first_activation)
@test isequal(second_activation, esn.second_activation)
@test isequal(first_lambda, esn.first_lambda)
@test isequal(second_lambda, esn.second_lambda)
@test isequal(nonlin_alg, esn.nonlin_alg)
@test size(esn.W) == (esn.res_size, esn.res_size)
@test size(esn.W_in) == (esn.res_size, esn.in_size)
@test size(esn.states) == (esn.res_size, train_len)

#test train
W_out = dafESNtrain(esn)
@test size(W_out) == (out_size, esn.res_size)
#test predict
output = dafESNpredict(esn, predict_len, W_out)
@test size(output) == (out_size, predict_len)

#test non linear algos
nla = [NonLinAlgT1, NonLinAlgT2, NonLinAlgT3]
for t in nla
    nonlin_alg = t
    esn = dafESN(approx_res_size,
            train,
            degree,
            radius,
            first_lambda,
            second_lambda,
            first_activation,
            second_activation,
            sigma,
            alpha,
            beta,
            nonlin_alg)
        
    W_out = dafESNtrain(esn)
    @test size(W_out) == (out_size, esn.res_size)
    output = dafESNpredict(esn, predict_len, W_out)
    @test size(output) == (out_size, predict_len)
end
 
