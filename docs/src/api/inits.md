# Echo State Networks Initializers

This page lists all initializers available in `ReservoirComputing.jl`.  
Clicking on any initializer name will take you to its dedicated documentation page,
where full details and examples are provided.

## Input layers

- [`chebyshev_mapping`](inits/chebyshev_mapping.md): Creates an input matrix
    using sine initialization followed by Chebyshev iterative mapping.
- [`informed_init`](inits/informed_init.md): Builds an informed ESN input
    layer allocating input vs. model channels based on γ-split.
- [`logistic_mapping`](inits/logistic_mapping.md): Generates an input
    layer using sine initialization followed by logistic-map recursion.
- [`minimal_init`](inits/minimal_init.md): Creates a uniform-weight input
    layer with signs determined by a sampling scheme.
- [`modified_lm`](inits/modified_lm.md): Builds an input-expanding
    logistic-map chain for each input dimension.
- [`scaled_rand`](inits/scaled_rand.md): Produces a uniformly scaled random
    input matrix with per-column or global scaling.
- [`weighted_init`](inits/weighted_init.md): Creates a block-structured
    weighted input layer with random weights per block.
- [`weighted_minimal`](inits/weighted_minimal.md): Generates a deterministic
    block-structured weighted input layer with optional sign sampling.

## Reservoirs

- [`block_diagonal`](inits/block_diagonal.md): Constructs a block-diagonal
    reservoir with constant-valued square blocks.
- [`chaotic_init`](inits/chaotic_init.md): Generates a reservoir from a digital
    chaotic adjacency graph with rescaled spectral radius.
- [`cycle_jumps`](inits/cycle_jumps.md): Builds a cycle reservoir augmented
    with periodic jump connections.
- [`delay_line`](inits/delay_line.md): Creates a delay-line reservoir using
    fixed offsets from the diagonal.
- [`delayline_backward`](inits/delayline_backward.md): Produces a delay-line
    reservoir with additional backward (feedback) connections.
- [`double_cycle`](inits/double_cycle.md): Creates two interlaced directed
    cycles (upper & lower) with independent weights.
- [`forward_connection`](inits/forward_connection.md): Builds a reservoir
    where each node connects forward by two steps.
- [`low_connectivity`](inits/low_connectivity.md): Creates a low-degree
    random (or enforced cycle) connectivity reservoir.
- [`pseudo_svd`](inits/pseudo_svd.md): Builds a reservoir by iteratively
    perturbing a diagonal matrix using pseudo-SVD rotations.
- [`rand_sparse`](inits/rand_sparse.md): Generates a random sparse
    reservoir with controlled sparsity and spectral radius.
- [`selfloop_cycle`](inits/selfloop_cycle.md): Builds a simple cycle
    reservoir enhanced with self-loops on all nodes.
- [`selfloop_delayline_backward`](inits/selfloop_delayline_backward.md): Combines delay
    line, self-loops, and backward offsets into one architecture.
- [`selfloop_backward_cycle`](inits/selfloop_backward_cycle.md): Creates a
    cycle where odd nodes self-loop and even nodes have forward/backward links.
- [`selfloop_forwardconnection`](inits/selfloop_forwardconnection.md): Adds self-loops
    onto a forward-connection reservoir (stride-2).
- [`simple_cycle`](inits/simple_cycle.md): Builds a basic directed ring
    reservoir with uniform weights.
- [`true_doublecycle`](inits/true_doublecycle.md): Constructs two overlapping
    cycles (forward + backward) using Rodan-style cycle rules.

## Building functions

- [`add_jumps!`](inits/add_jumps!.md): Inserts jump connections at
    fixed intervals into an existing reservoir.
- [`backward_connection!`](inits/backward_connection!.md): Adds backward
    (feedback) connections at a fixed shift.
- [`delay_line!`](inits/delay_line!.md): Writes delay connections into
    an existing matrix at a specified diagonal offset.
- [`reverse_simple_cycle!`](inits/reverse_simple_cycle!.md): Adds a
    reversed directed cycle (descending indices) to a matrix.
- [`scale_radius!`](inits/scale_radius!.md): Rescales a reservoir
    matrix to match a target spectral radius.
- [`self_loop!`](inits/self_loop!.md): Adds self-loop weights
    along the diagonal of an existing matrix.
- [`simple_cycle!`](inits/simple_cycle!.md): Writes a directed
    cycle pattern into a preallocated reservoir matrix.
