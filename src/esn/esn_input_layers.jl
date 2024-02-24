abstract type AbstractLayer end

struct WeightedLayer{T} <: AbstractLayer
    scaling::T
end

"""
    WeightedInput(scaling)
    WeightedInput(;scaling=0.1)

Creates a `WeightedInput` layer initializer for Echo State Networks.
This initializer generates a weighted input matrix with random non-zero
elements distributed uniformly within the range [-`scaling`, `scaling`],
following the approach in [^Lu].

# Parameters

  - `scaling`: The scaling factor for the weight distribution (default: 0.1).

# Returns

  - A `WeightedInput` instance to be used for initializing the input layer of an ESN.

Reference:

[^Lu]: Lu, Zhixin, et al.
    "Reservoir observers: Model-free inference of unmeasured variables in chaotic systems."
    Chaos: An Interdisciplinary Journal of Nonlinear Science 27.4 (2017): 041102.
"""
function WeightedLayer(; scaling = 0.1)
    return WeightedLayer(scaling)
end

function create_layer(input_layer::WeightedLayer,
        approx_res_size,
        in_size;
        matrix_type = Matrix{Float64})
    scaling = input_layer.scaling
    res_size = Int(floor(approx_res_size / in_size) * in_size)
    layer_matrix = zeros(res_size, in_size)
    q = floor(Int, res_size / in_size)

    for i in 1:in_size
        layer_matrix[((i - 1) * q + 1):((i) * q), i] = rand(Uniform(-scaling, scaling), 1,
            q)
    end

    return Adapt.adapt(matrix_type, layer_matrix)
end

function create_layer(layer, args...; kwargs...)
    return layer
end

"""
    DenseLayer(scaling)
    DenseLayer(;scaling=0.1)

Creates a `DenseLayer` initializer for Echo State Networks, generating a fully connected input layer.
The layer is initialized with random weights uniformly distributed within [-`scaling`, `scaling`].
This scaling factor can be provided either as an argument or a keyword argument.
The `DenseLayer` is the default input layer in `ESN` construction.

# Parameters

  - `scaling`: The scaling factor for weight distribution (default: 0.1).

# Returns

  - A `DenseLayer` instance for initializing the ESN's input layer.
"""
struct DenseLayer{T} <: AbstractLayer
    scaling::T
end

function DenseLayer(; scaling = 0.1)
    return DenseLayer(scaling)
end

"""
    create_layer(input_layer::AbstractLayer, res_size, in_size)

Generates a matrix layer of size `res_size` x `in_size`, constructed according to the specifications of the `input_layer`.

# Parameters

  - `input_layer`: An instance of `AbstractLayer` determining the layer construction.
  - `res_size`: The number of rows (reservoir size) for the layer.
  - `in_size`: The number of columns (input size) for the layer.

# Returns

  - A matrix representing the constructed layer.
"""
function create_layer(input_layer::DenseLayer,
        res_size,
        in_size;
        matrix_type = Matrix{Float64})
    scaling = input_layer.scaling
    layer_matrix = rand(Uniform(-scaling, scaling), res_size, in_size)
    return Adapt.adapt(matrix_type, layer_matrix)
end

"""
    SparseLayer(scaling, sparsity)
    SparseLayer(scaling; sparsity=0.1)
    SparseLayer(;scaling=0.1, sparsity=0.1)

Creates a `SparseLayer` initializer for Echo State Networks, generating a sparse input layer.
The layer is initialized with weights distributed within [-`scaling`, `scaling`]
and a specified `sparsity` level. Both `scaling` and `sparsity` can be set as arguments or keyword arguments.

# Parameters

  - `scaling`: Scaling factor for weight distribution (default: 0.1).
  - `sparsity`: Sparsity level of the layer (default: 0.1).

# Returns

  - A `SparseLayer` instance for initializing ESN's input layer with sparse connections.
"""
struct SparseLayer{T} <: AbstractLayer
    scaling::T
    sparsity::T
end

function SparseLayer(; scaling = 0.1, sparsity = 0.1)
    return SparseLayer(scaling, sparsity)
end

function SparseLayer(scaling_arg; scaling = scaling_arg, sparsity = 0.1)
    return SparseLayer(scaling, sparsity)
end

function create_layer(input_layer::SparseLayer,
        res_size,
        in_size;
        matrix_type = Matrix{Float64})
    layer_matrix = Matrix(sprand(res_size, in_size, input_layer.sparsity))
    layer_matrix = 2.0 .* (layer_matrix .- 0.5)
    replace!(layer_matrix, -1.0 => 0.0)
    layer_matrix = input_layer.scaling .* layer_matrix
    return Adapt.adapt(matrix_type, layer_matrix)
end

#from "minimum complexity echo state network" Rodan
#and "simple deterministically constructed cycle reservoirs with regular jumps"
#by Rodan and Tino
struct BernoulliSample{T}
    p::T
end

"""
    BernoulliSample(p)
    BernoulliSample(;p=0.5)

Creates a `BernoulliSample` constructor for the `MinimumLayer`.
It uses a Bernoulli distribution to determine the sign of weights in the input layer.
The parameter `p` sets the probability of a weight being positive, as per the `Distributions` package.
This method of sign weight determination for input layers is based on the approach in [^Rodan].

# Parameters

  - `p`: Probability of a positive weight (default: 0.5).

# Returns

  - A `BernoulliSample` instance for generating sign weights in `MinimumLayer`.

Reference:

[^Rodan]: Rodan, Ali, and Peter Tino.
    "Minimum complexity echo state network."
    IEEE Transactions on Neural Networks 22.1 (2010): 131-144.
"""
function BernoulliSample(; p = 0.5)
    return BernoulliSample(p)
end

struct IrrationalSample{K}
    irrational::Irrational
    start::K
end

"""
    IrrationalSample(irrational, start)
    IrrationalSample(;irrational=pi, start=1)

Creates an `IrrationalSample` constructor for the `MinimumLayer`.
It determines the sign of weights in the input layer based on the decimal expansion of an `irrational` number.
The `start` parameter sets the starting point in the decimal sequence.
The signs are assigned based on the thresholding of each decimal digit against 4.5, as described in [^Rodan].

# Parameters

  - `irrational`: An irrational number for weight sign determination (default: π).
  - `start`: Starting index in the decimal expansion (default: 1).

# Returns

  - An `IrrationalSample` instance for generating sign weights in `MinimumLayer`.

Reference:

[^Rodan]: Rodan, Ali, and Peter Tiňo.
    "Simple deterministically constructed cycle reservoirs with regular jumps."
    Neural Computation 24.7 (2012): 1822-1852.
"""
function IrrationalSample(; irrational = pi, start = 1)
    return IrrationalSample(irrational, start)
end

struct MinimumLayer{T, K} <: AbstractLayer
    weight::T
    sampling::K
end

"""
    MinimumLayer(weight, sampling)
    MinimumLayer(weight; sampling=BernoulliSample(0.5))
    MinimumLayer(;weight=0.1, sampling=BernoulliSample(0.5))

Creates a `MinimumLayer` initializer for Echo State Networks, generating a fully connected input layer.
This layer has a uniform absolute weight value (`weight`) with the sign of each
weight determined by the `sampling` method. This approach, as detailed in [^Rodan1] and [^Rodan2],
allows for controlled weight distribution in the layer.

# Parameters

  - `weight`: Absolute value of weights in the layer.
  - `sampling`: Method for determining the sign of weights (default: `BernoulliSample(0.5)`).

# Returns

  - A `MinimumLayer` instance for initializing the ESN's input layer.

References:

[^Rodan1]: Rodan, Ali, and Peter Tino.
    "Minimum complexity echo state network."
    IEEE Transactions on Neural Networks 22.1 (2010): 131-144.
[^Rodan2]: Rodan, Ali, and Peter Tiňo.
    "Simple deterministically constructed cycle reservoirs with regular jumps."
    Neural Computation 24.7 (2012): 1822-1852.
"""
function MinimumLayer(weight; sampling = BernoulliSample(0.5))
    return MinimumLayer(weight, sampling)
end

function MinimumLayer(; weight = 0.1, sampling = BernoulliSample(0.5))
    return MinimumLayer(weight, sampling)
end

function create_layer(input_layer::MinimumLayer,
        res_size,
        in_size;
        matrix_type = Matrix{Float64})
    sampling = input_layer.sampling
    weight = input_layer.weight
    layer_matrix = create_minimum_input(sampling, res_size, in_size, weight)
    return Adapt.adapt(matrix_type, layer_matrix)
end

function create_minimum_input(sampling::BernoulliSample, res_size, in_size, weight)
    p = sampling.p
    input_matrix = zeros(res_size, in_size)
    for i in 1:res_size
        for j in 1:in_size
            rand(Bernoulli(p)) ? input_matrix[i, j] = weight : input_matrix[i, j] = -weight
        end
    end

    return input_matrix
end

function create_minimum_input(sampling::IrrationalSample, res_size, in_size, weight)
    setprecision(BigFloat, Int(ceil(log2(10) * (res_size * in_size + sampling.start + 1))))
    ir_string = string(BigFloat(sampling.irrational)) |> collect
    deleteat!(ir_string, findall(x -> x == '.', ir_string))
    ir_array = zeros(length(ir_string))
    input_matrix = zeros(res_size, in_size)

    for i in 1:length(ir_string)
        ir_array[i] = parse(Int, ir_string[i])
    end

    co = sampling.start
    counter = 1

    for i in 1:res_size
        for j in 1:in_size
            ir_array[counter] < 5 ? input_matrix[i, j] = -weight :
            input_matrix[i, j] = weight
            counter += 1
        end
    end

    return input_matrix
end

struct InformedLayer{T, K, M} <: AbstractLayer
    scaling::T
    gamma::K
    model_in_size::M
end

"""
    InformedLayer(model_in_size; scaling=0.1, gamma=0.5)

Creates an `InformedLayer` initializer for Echo State Networks (ESNs) that generates
a weighted input layer matrix. The matrix contains random non-zero elements drawn from
the range [-`scaling`, `scaling`]. This initializer ensures that a fraction (`gamma`)
of reservoir nodes are exclusively connected to the raw inputs, while the rest are
connected to the outputs of a prior knowledge model, as described in [^Pathak].

# Arguments

  - `model_in_size`: The size of the prior knowledge model's output,
    which determines the number of columns in the input layer matrix.

# Keyword Arguments

  - `scaling`: The absolute value of the weights (default: 0.1).
  - `gamma`: The fraction of reservoir nodes connected exclusively to raw inputs (default: 0.5).

# Returns

  - An `InformedLayer` instance for initializing the ESN's input layer matrix.

Reference:

[^Pathak]: Jaideep Pathak et al.
    "Hybrid Forecasting of Chaotic Processes: Using Machine Learning in Conjunction with a Knowledge-Based Model" (2018).
"""
function InformedLayer(model_in_size; scaling = 0.1, gamma = 0.5)
    return InformedLayer(scaling, gamma, model_in_size)
end

function create_layer(input_layer::InformedLayer,
        res_size,
        in_size;
        matrix_type = Matrix{Float64})
    scaling = input_layer.scaling
    state_size = in_size - input_layer.model_in_size

    if state_size <= 0
        throw(DimensionMismatch("in_size must be greater than model_in_size"))
    end

    input_matrix = zeros(res_size, in_size)
    #Vector used to find res nodes not yet connected
    zero_connections = zeros(in_size)
    #Num of res nodes allotted for raw states
    num_for_state = floor(Int, res_size * input_layer.gamma)
    #Num of res nodes allotted for prior model input
    num_for_model = floor(Int, (res_size * (1 - input_layer.gamma)))

    for i in 1:num_for_state
        #find res nodes with no connections
        idxs = findall(Bool[zero_connections == input_matrix[i, :]
                            for i in 1:size(input_matrix, 1)])
        random_row_idx = idxs[rand(1:end)]
        random_clm_idx = range(1, state_size, step = 1)[rand(1:end)]
        input_matrix[random_row_idx, random_clm_idx] = rand(Uniform(-scaling, scaling))
    end

    for i in 1:num_for_model
        idxs = findall(Bool[zero_connections == input_matrix[i, :]
                            for i in 1:size(input_matrix, 1)])
        random_row_idx = idxs[rand(1:end)]
        random_clm_idx = range(state_size + 1, in_size, step = 1)[rand(1:end)]
        input_matrix[random_row_idx, random_clm_idx] = rand(Uniform(-scaling, scaling))
    end

    return Adapt.adapt(matrix_type, input_matrix)
end

"""
    NullLayer()

Creates a `NullLayer` initializer for Echo State Networks (ESNs) that generates a vector of zeros.

# Returns

  - A `NullLayer` instance for initializing the ESN's input layer matrix.
"""
struct NullLayer <: AbstractLayer end

function create_layer(input_layer::NullLayer,
        res_size,
        in_size;
        matrix_type = Matrix{Float64})
    return Adapt.adapt(matrix_type, zeros(res_size, in_size))
end
