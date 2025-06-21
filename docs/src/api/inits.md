# Echo State Networks Initializers

## Input layers

```@docs
    chebyshev_mapping
    informed_init
    logistic_mapping
    minimal_init
    modified_lm
    scaled_rand
    weighted_init
    weighted_minimal
```

## Reservoirs

```@docs
    block_diagonal
    chaotic_init
    cycle_jumps
    delay_line
    delay_line_backward
    double_cycle
    forward_connection
    low_connectivity
    pseudo_svd
    rand_sparse
    selfloop_cycle
    selfloop_delayline_backward
    selfloop_feedback_cycle
    selfloop_forward_connection
    simple_cycle
    true_double_cycle
```

## Building functions

```@docs
    add_jumps!
    backward_connection!
    delay_line!
    reverse_simple_cycle!
    scale_radius!
    self_loop!
    simple_cycle!
```

## References

```@bibliography
Pages = ["inits.md"]
Canonical = false
```