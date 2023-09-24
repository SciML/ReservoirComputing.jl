abstract type AbstractLayer end

struct WeightedLayer{T} <: AbstractLayer
    scaling::T
end

"""
    WeightedInput(scaling)
    WeightedInput(;scaling=0.1)

Returns a weighted layer initializer object, that will produce a weighted input matrix with
random non-zero elements drawn from [-```scaling```, ```scaling```], as described
in [1]. The ```scaling``` factor can be given as arg or kwarg.

[1] Lu, Zhixin, et al. "_Reservoir observers: Model-free inference of unmeasured variables
in chaotic systems._"
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

Returns a fully connected layer initializer object, that will produce a weighted input
matrix with random non-zero elements drawn from [-```scaling```, ```scaling```].
The ```scaling``` factor can be given as arg or kwarg. This is the default choice in the
```ESN``` construction.
"""
struct DenseLayer{T} <: AbstractLayer
    scaling::T
end

function DenseLayer(; scaling = 0.1)
    return DenseLayer(scaling)
end

"""
    create_layer(input_layer::AbstractLayer, res_size, in_size)

Returns a ```res_size``` times ```in_size``` matrix layer, built according to the
```input_layer``` constructor.
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

Returns a sparsely connected layer initializer object, that will produce a random sparse
input matrix with random non-zero elements drawn from [-```scaling```, ```scaling```] and
given sparsity. The ```scaling``` and ```sparsity``` factors can be given as args or kwargs.
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

Returns a Bernoulli sign constructor for the ```MinimumLayer``` call. The ```p``` factor
determines the probability of the result, as in the Distributions call. The value can be
passed as an arg or kwarg. This sign weight determination for input layers is introduced
in [1].

[1] Rodan, Ali, and Peter Tino. "_Minimum complexity echo state network._"
IEEE transactions on neural networks 22.1 (2010): 131-144.
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

Returns an irrational sign constructor for the ```MinimumLayer``` call. The values can be
passed as args or kwargs. The sign of the weight is decided from the decimal expansion of
the given ```irrational```. The first ```start``` decimal digits are thresholded at 4.5,
then the n-th input sign will be + and - respectively.

[1] Rodan, Ali, and Peter Tiňo. "_Simple deterministically constructed cycle reservoirs
with regular jumps._" Neural computation 24.7 (2012): 1822-1852.
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

Returns a fully connected layer initializer object. The matrix constructed with this
initializer presents the same absolute weight value, decided by the ```weight``` factor.
The sign of each entry is decided by the ```sampling``` struct. Construction detailed
in [1] and [2].

[1] Rodan, Ali, and Peter Tino. "_Minimum complexity echo state network._"
IEEE transactions on neural networks 22.1 (2010): 131-144.
[2] Rodan, Ali, and Peter Tiňo. "_Simple deterministically constructed cycle reservoirs
with regular jumps._" Neural computation 24.7 (2012): 1822-1852.
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

Returns a weighted input layer matrix, with random non-zero elements drawn from
[-```scaling```, ```scaling```], where some γ of reservoir nodes are connected exclusively
to the raw inputs, and the rest to the outputs of the prior knowledge model,
as described in [1].

[1] Jaideep Pathak et al. "_Hybrid Forecasting of Chaotic Processes: Using Machine Learning
in Conjunction with a Knowledge-Based Model_" (2018)
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
    NullLayer(model_in_size; scaling=0.1, gamma=0.5)

Returns a vector of zeros.
"""
struct NullLayer <: AbstractLayer end

function create_layer(input_layer::NullLayer,
                      res_size,
                      in_size;
                      matrix_type = Matrix{Float64})
    return Adapt.adapt(matrix_type, zeros(res_size, in_size))
end
