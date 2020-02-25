using ReservoirComputing  
using ParameterizedFunctions
using DifferentialEquations

#lorenz system parameters
u0 = [1.0,0.0,0.0]                       
tspan = (0.0,200.0)                      
p = [10.0,28.0,8/3]
#model parameters
shift = 1
approx_res_size = 300
N = 3
radius = 1.2
degree = 6
sigma = 0.1
in_size = N
out_size = N
train_len = 5000
predict_len = 1250
beta = 0.0
alpha = 1.0
nonlin_alg = "None"

#define lorenz system 
function lorenz(du,u,p,t)
    du[1] = p[1]*(u[2]-u[1])
    du[2] = u[1]*(p[2]-u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
end
    
prob = ODEProblem(lorenz, u0, tspan, p)  
sol = solve(prob, AB4(), dt=0.02)   
v = sol.u
data = Matrix(hcat(v...))
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
