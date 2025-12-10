# Next Generation Reservoir Computing

This tutorial shows how to use next generation reservoir computing [NGRC](@ref)
in ReservoirComputing.jl to model the chaotic Lorenz system.

NGRC works differently compared to traditional reservoir computing. In NGRC
the reservoir is replaced with:
  - A delay embedding of the input
  - A nonlinear feature map
The model is finally trained through ridge regression, like a normal RC.


In this tutorial we will :
  - simulate the Lorenz system,
  - build an NGRC model with delayed inputs and polynomial features, following the
    [original paper](https://doi.org/10.1038/s41467-021-25801-2),
  - train it on one-step increments,
  - roll it out generatively and compare with the true trajectory.

## 1. Setup and imports

First we need to load the necessary packages. We are going to use the following:

```@example ngrc
using OrdinaryDiffEq
using Random
using ReservoirComputing
using Plots
using Statistics
```

## 2. Define Lorenz system and generate data

We define the Lorenz system and integrate it to generate a long trajectory:

```@example ngrc
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

prob = ODEProblem(
    lorenz!,
    Float32[1.0, 0.0, 0.0],
    (0.0, 200.0),
    (10.0f0, 28.0f0, 8/3f0),
)

data = Array(solve(prob, ABM54(); dt = 0.025))  # size: (3, T)
```

We then split the time series into training and testing segments:

```@example ngrc
shift = 300
train_len = 500
predict_len = 900

input_data = data[:, shift:(shift + train_len - 1)]
target_data = data[:, (shift + 1):(shift + train_len)]
test_data = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]
```

## 3. Normalization

It is good practice to normalize the data, especially for polynomial features:

```@example ngrc
in_mean = mean(input_data; dims = 2)
in_std = std(input_data;  dims = 2)

train_norm_x = (input_data  .- in_mean) ./ in_std
train_norm_y = (target_data .- in_mean) ./ in_std
test_norm_x  = (test_data   .- in_mean) ./ in_std

# We train an increment (residual) model: Δy = y_{t+1} − y_t
train_delta_y = train_norm_y .- train_norm_x
```

## 4. Build the NGRC model

Now that we have the data we can start building the model.
Following the approach of the paper we first define two feature functions:

  - a constant feature
  - a second order polynomial monomial 
  
```@example ngrc
const_feature = x -> Float32[1.0]
poly_feature  = x -> polynomial_monomials(x; degrees = 1:2)
```

Finally, we can construct the NGRC model.

We set the following:

```@example ngrc
in_dims = 3
out_dims = 3
num_delays = 1
```

With `in_dims=3` and `num_delays=1` the delayed input length is 6.
Adding the polinomial of degrees 1 and 2 will put give us 21 more. Finally, the constant
term adds 1 more feature. In total we have 28 features. 

We can pass the number of features to `ro_dims` to initialize the [`LinearReadout`](@ref)
with the correct dimensions. However, unless one is planning to fry run the model without training,
the [`train`](@ref) function will take care to adjust the dimensions.

Now we build the NGRC:

```@example ngrc
rng = MersenneTwister(0)

ngrc = NGRC(in_dims, out_dims; num_delays = num_delays, stride = 1, features = (const_feature, poly_feature),
    include_input = false,  # we already encode everything in the features
    ro_dims = 28,
    readout_activation = identity)

ps, st = setup(rng, ngrc)
```

At this point, `ngrc` is a fully specified model with:
  - a [`DelayLayer`](@ref) that builds a 6-dimensional delayed vector from the 3D input,
  - a [`NonlinearFeaturesLayer`](@ref) that maps that vector to 28 polynomial features,
  - a [`LinearReadout`](@ref) (28 => 3).

## 5. Training the NGRC readout

We now train the linear readout using ridge regression on the increment `train_delta_y`:

```@example ngrc
ps, st = train!(ngrc, train_norm_x, train_delta_y, ps, st;
    train_method = StandardRidge(2.5e-6))
```

where [`StandardRidge`](@ref) is the ridge regression provided natively by ReservoirComputing.jl.

## 6. Generative prediction

We now perform generative prediction on the increments to obtain the predicted time series:

```@example ngrc
single_step = copy(test_norm_x[:, 1]) # normalized initial condition
traj_norm = similar(test_norm_x, 3, predict_len)

for step in 1:predict_len
    global st
    delta_step, st = ngrc(single_step, ps, st)
    single_step .= single_step .+ delta_step # increment update in normalized space
    traj_norm[:, step] .= single_step
end
```

Finally, we unscale back to the original coordinates:

```@example ngrc
traj = traj_norm .* in_std .+ in_mean # size: (3, predict_len)
```

## 7. Visualization

We can now compare the predicted trajectory with the true Lorenz data on the test segment:

```@example ngrc
plot(transpose(test_data)[:, 1], transpose(test_data)[:, 2], transpose(test_data)[:, 3]; label="actual");
plot!(transpose(traj)[:, 1], transpose(traj)[:, 2], transpose(traj)[:, 3]; label="predicted")
```
