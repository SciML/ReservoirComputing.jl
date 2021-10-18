using ReservoirComputing

const res_size = 10
const in_size = 3
const scaling = 0.1
const weight = 0.2

#testing WeightedInput implicit and esplicit constructors
input_constructor = WeightedInput(scaling)
input_matrix = create_input_layer(res_size, in_size, input_constructor)
@test size(input_matrix) == (Int(floor(res_size/in_size)*in_size), in_size)
@test maximum(input_matrix) <= scaling

input_constructor = WeightedInput(scaling=scaling)
input_matrix = create_input_layer(res_size, in_size, input_constructor)
@test size(input_matrix) == (Int(floor(res_size/in_size)*in_size), in_size)
@test maximum(input_matrix) <= scaling

#testing DenseInput implicit and esplicit constructors
input_constructor = DenseInput(scaling)
input_matrix = create_input_layer(res_size, in_size, input_constructor)
@test size(input_matrix) == (res_size, in_size)
@test maximum(input_matrix) <= scaling

input_constructor = DenseInput(scaling=scaling)
input_matrix = create_input_layer(res_size, in_size, input_constructor)
@test size(input_matrix) == (res_size, in_size)
@test maximum(input_matrix) <= scaling

#testing SparseInput implicit and esplicit constructors
input_constructor = SparseInput(scaling)
input_matrix = create_input_layer(res_size, in_size, input_constructor)
@test size(input_matrix) == (res_size, in_size)
@test maximum(input_matrix) <= scaling

input_constructor = SparseInput(scaling=scaling)
input_matrix = create_input_layer(res_size, in_size, input_constructor)
@test size(input_matrix) == (res_size, in_size)
@test maximum(input_matrix) <= scaling

#testing MinimumInput implicit and esplicit constructors
input_constructor = MinimumInput(weight)
input_matrix = create_input_layer(res_size, in_size, input_constructor)
@test size(input_matrix) == (res_size, in_size)
@test maximum(input_matrix) == weight

input_constructor = MinimumInput(weight=weight)
input_matrix = create_input_layer(res_size, in_size, input_constructor)
@test size(input_matrix) == (res_size, in_size)
@test maximum(input_matrix) == weight