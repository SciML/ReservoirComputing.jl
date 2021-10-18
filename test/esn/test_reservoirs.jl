using ReservoirComputing

const res_size = 10
const radius = 1.0
const sparsity = 0.1
const weight = 0.2
const jump_size = 3

#testing RandSparseReservoir implicit and esplicit constructors
reservoir_constructor = RandSparseReservoir(radius, sparsity)
reservoir_matrix = create_reservoir(res_size, reservoir_constructor)
@test size(reservoir_matrix) == (res_size, res_size)

reservoir_constructor = RandSparseReservoir(radius=radius, sparsity=sparsity)
reservoir_matrix = create_reservoir(res_size, reservoir_constructor)
@test size(reservoir_matrix) == (res_size, res_size)

#testing PseudoSVDReservoir implicit and esplicit constructors
reservoir_constructor = PseudoSVDReservoir(radius, sparsity)
reservoir_matrix = create_reservoir(res_size, reservoir_constructor)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) <= radius

reservoir_constructor = PseudoSVDReservoir(max_value=radius, sparsity=sparsity)
reservoir_matrix = create_reservoir(res_size, reservoir_constructor)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) <= radius

#testing DelayLineReservoir implicit and esplicit constructors
reservoir_constructor = DelayLineReservoir(weight)
reservoir_matrix = create_reservoir(res_size, reservoir_constructor)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight

reservoir_constructor = DelayLineReservoir(weight=weight)
reservoir_matrix = create_reservoir(res_size, reservoir_constructor)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight

#testing DelayLineReservoir implicit and esplicit constructors
reservoir_constructor = DelayLineBackwardReservoir(weight, weight)
reservoir_matrix = create_reservoir(res_size, reservoir_constructor)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight

reservoir_constructor = DelayLineBackwardReservoir(weight=weight, fb_weight=weight)
reservoir_matrix = create_reservoir(res_size, reservoir_constructor)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight

#testing SimpleCycleReservoir implicit and esplicit constructors
reservoir_constructor = SimpleCycleReservoir(weight)
reservoir_matrix = create_reservoir(res_size, reservoir_constructor)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight

reservoir_constructor = SimpleCycleReservoir(weight=weight)
reservoir_matrix = create_reservoir(res_size, reservoir_constructor)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight

#testing CycleJumpsReservoir implicit and esplicit constructors
reservoir_constructor = CycleJumpsReservoir(weight, weight, jump_size)
reservoir_matrix = create_reservoir(res_size, reservoir_constructor)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight

reservoir_constructor = CycleJumpsReservoir(cycle_weight=weight, jump_weight=weight, jump_size=jump_size)
reservoir_matrix = create_reservoir(res_size, reservoir_constructor)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight