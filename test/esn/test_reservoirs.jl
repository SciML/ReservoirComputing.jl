using ReservoirComputing

const res_size = 20
const radius = 1.0
const sparsity = 0.1
const weight = 0.2
const jump_size = 3

#testing RandSparseReservoir implicit and esplicit constructors
reservoir_constructor = RandSparseReservoir(res_size, radius, sparsity)
reservoir_matrix = create_reservoir(reservoir_constructor, res_size)
@test size(reservoir_matrix) == (res_size, res_size)

reservoir_constructor = RandSparseReservoir(res_size, radius=radius, sparsity=sparsity)
reservoir_matrix = create_reservoir(reservoir_constructor, res_size)
@test size(reservoir_matrix) == (res_size, res_size)

#testing PseudoSVDReservoir implicit and esplicit constructors
reservoir_constructor = PseudoSVDReservoir(res_size, radius, sparsity)
reservoir_matrix = create_reservoir(reservoir_constructor, res_size)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) <= radius

reservoir_constructor = PseudoSVDReservoir(res_size, max_value=radius, sparsity=sparsity)
reservoir_matrix = create_reservoir(reservoir_constructor, res_size)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) <= radius

#testing DelayLineReservoir implicit and esplicit constructors
reservoir_constructor = DelayLineReservoir(res_size, weight)
reservoir_matrix = create_reservoir(reservoir_constructor, res_size)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight

reservoir_constructor = DelayLineReservoir(res_size, weight=weight)
reservoir_matrix = create_reservoir(reservoir_constructor, res_size)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight

#testing DelayLineReservoir implicit and esplicit constructors
reservoir_constructor = DelayLineBackwardReservoir(res_size, weight, weight)
reservoir_matrix = create_reservoir(reservoir_constructor, res_size)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight

reservoir_constructor = DelayLineBackwardReservoir(res_size, weight=weight, fb_weight=weight)
reservoir_matrix = create_reservoir(reservoir_constructor, res_size)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight

#testing SimpleCycleReservoir implicit and esplicit constructors
reservoir_constructor = SimpleCycleReservoir(res_size, weight)
reservoir_matrix = create_reservoir(reservoir_constructor, res_size)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight

reservoir_constructor = SimpleCycleReservoir(res_size, weight=weight)
reservoir_matrix = create_reservoir(reservoir_constructor, res_size)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight

#testing CycleJumpsReservoir implicit and esplicit constructors
reservoir_constructor = CycleJumpsReservoir(res_size, weight, weight, jump_size)
reservoir_matrix = create_reservoir(reservoir_constructor, res_size)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight

reservoir_constructor = CycleJumpsReservoir(res_size, cycle_weight=weight, jump_weight=weight, jump_size=jump_size)
reservoir_matrix = create_reservoir(reservoir_constructor, res_size)
@test size(reservoir_matrix) == (res_size, res_size)
@test maximum(reservoir_matrix) == weight

#testing NullReservoir constructors
reservoir_constructor = NullReservoir()
reservoir_matrix = create_reservoir(reservoir_constructor, res_size)
@test size(reservoir_matrix) == (res_size, res_size)
