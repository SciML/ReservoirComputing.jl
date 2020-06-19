using ReservoirComputing
using ParameterizedFunctions
using OrdinaryDiffEq
using Plots 

#rossler system parameters
u0 = [0.1,0.5,0.1]                       
tspan = (0.0,1000.0)                      
p = [0.5,2.0,4.0]

#define rossler system 
function rossler(du,u,p,t)
    du[1] = -u[2]-u[3]
    du[2] = u[1] + p[1]*u[2]
    du[3] = p[2]+u[3]*(u[1]-p[3])
end

#solve system
prob = ODEProblem(rossler, u0, tspan, p)  
sol = solve(prob, ABM54(), dt=0.02)   
v = sol.u
data = Matrix(hcat(v...))

#model parameters
shift = 500
approx_res_size = 300
radius = 1.15
activation = tanh
degree = 4
sigma = 0.1
train_len = 5000
predict_len = 2500
beta = 0.0
alpha = 1.0
nla_type = NLADefault()
extended_states = false

#get data
train = data[:, shift:shift+train_len-1]
test = data[:, shift+train_len:shift+train_len+predict_len-1]

#create echo state network  
esn = ESN(approx_res_size,
    train,
    degree,
    radius,
    activation = activation,
    sigma = sigma,
    alpha = alpha,
    nla_type = nla_type,
    extended_states = extended_states)


#training and prediction
W_out = ESNtrain(esn, beta)
output = ESNpredict(esn, predict_len, W_out)

#plots and images
comp = plot(transpose(output),layout=(3,1), label="predicted")
comp = plot!(transpose(test),layout=(3,1), label="actual")
savefig(comp, "rossler_coord")
sf = plot(transpose(output)[:,1], transpose(output)[:,2], transpose(output)[:,3], label="predicted")
sf2 = plot(transpose(test)[:,1], transpose(test)[:,2], transpose(test)[:,3], label="actual")
final = plot(sf, sf2, layout=(1,2), label = ["predicted" "actual"])
savefig(final, "rossler_attractor") 
