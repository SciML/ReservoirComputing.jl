using ReservoirComputing

const res_size = 10
const in_size = 3
const scaling = 0.1
const weight = 0.2

#testing WeightedLayer implicit and esplicit constructors
input_constructor = WeightedLayer(scaling)
input_matrix = create_layer(input_constructor, res_size, in_size)
@test size(input_matrix) == (Int(floor(res_size / in_size) * in_size), in_size)
@test maximum(input_matrix) <= scaling

input_constructor = WeightedLayer(scaling = scaling)
input_matrix = create_layer(input_constructor, res_size, in_size)
@test size(input_matrix) == (Int(floor(res_size / in_size) * in_size), in_size)
@test maximum(input_matrix) <= scaling

#testing DenseLayer implicit and esplicit constructors
input_constructor = DenseLayer(scaling)
input_matrix = create_layer(input_constructor, res_size, in_size)
@test size(input_matrix) == (res_size, in_size)
@test maximum(input_matrix) <= scaling

input_constructor = DenseLayer(scaling = scaling)
input_matrix = create_layer(input_constructor, res_size, in_size)
@test size(input_matrix) == (res_size, in_size)
@test maximum(input_matrix) <= scaling

#testing SparseLayer implicit and esplicit constructors
input_constructor = SparseLayer(scaling)
input_matrix = create_layer(input_constructor, res_size, in_size)
@test size(input_matrix) == (res_size, in_size)
@test maximum(input_matrix) <= scaling

input_constructor = SparseLayer(scaling = scaling)
input_matrix = create_layer(input_constructor, res_size, in_size)
@test size(input_matrix) == (res_size, in_size)
@test maximum(input_matrix) <= scaling

#testing MinimumLayer implicit and esplicit constructors
input_constructor = MinimumLayer(weight)
input_matrix = create_layer(input_constructor, res_size, in_size)
@test size(input_matrix) == (res_size, in_size)
@test maximum(input_matrix) == weight

input_constructor = MinimumLayer(weight = weight)
input_matrix = create_layer(input_constructor, res_size, in_size)
@test size(input_matrix) == (res_size, in_size)
@test maximum(input_matrix) == weight

#testing NullLayer constructor
input_constructor = NullLayer()
input_matrix = create_layer(input_constructor, res_size, in_size)
@test size(input_matrix) == (res_size, in_size)
