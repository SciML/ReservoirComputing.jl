using ReservoirComputing
using ParameterizedFunctions
using OrdinaryDiffEq
using Plots 

#lorenz system parameters
u0 = [1.0,0.0,0.0]                       
tspan = (0.0,1000.0)                      
p = [10.0,28.0,8/3]

#define lorenz system 
function lorenz(du,u,p,t)
    du[1] = p[1]*(u[2]-u[1])
    du[2] = u[1]*(p[2]-u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
end

#solve system
prob = ODEProblem(lorenz, u0, tspan, p)  
sol = solve(prob, ABM54(), dt=0.02)   
v = sol.u
data = Matrix(hcat(v...))

#model parameters
shift = 300
approx_res_size = 300
radius = 1.2
activation = atan
degree = 6
sigma = 0.1
train_len = 5000
predict_len = 1250
beta = 0.0
alpha = 1.0
nonlin_alg = NonLinAlgT2

#get data
train = data[:, shift:shift+train_len-1]
test = data[:, shift+train_len:shift+train_len+predict_len-1]

#create echo state network  
esn = ESN(approx_res_size,
    train,
    degree,
    radius,
    activation,
    sigma,
    alpha,
    beta,
    nonlin_alg)

#training and prediction
W_out = ESNtrain(esn)
output = ESNpredict(esn, predict_len, W_out)

#plots and images
comp = plot(transpose(output),layout=(3,1), label="predicted")
comp = plot!(transpose(test),layout=(3,1), label="actual")
savefig(comp, "lorenz_coord")
sf = plot(transpose(output)[:,1], transpose(output)[:,2], transpose(output)[:,3], label="predicted")
sf2 = plot(transpose(test)[:,1], transpose(test)[:,2], transpose(test)[:,3], label="actual")
final = plot(sf, sf2, layout=(1,2), label = ["predicted" "actual"])
savefig(final, "lorenz_attractor")
