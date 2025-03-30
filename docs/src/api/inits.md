# Echo State Networks Initializers

## Input layers

```@docs
    scaled_rand
    weighted_init
    minimal_init
    weighted_minimal
    chebyshev_mapping
    logistic_mapping
    modified_lm
    informed_init
```

## Reservoirs

```@docs
    rand_sparse
    pseudo_svd
    chaotic_init
    low_connectivity
    delay_line
    delay_line_backward
    simple_cycle
    cycle_jumps
    double_cycle
    true_double_cycle
    selfloop_cycle
    selfloop_feedback_cycle
    selfloop_delayline_backward
    selfloop_forward_connection
    forward_connection
```

## Building functions

```@docs
    scale_radius!
    delay_line!
    backward_connection!
    simple_cycle!
    reverse_simple_cycle!
    self_loop!
    add_jumps!
```
