# Echo State Networks Initializers

## Input layers

```@docs
    scaled_rand
    weighted_init
    weighted_minimal
    informed_init
    minimal_init
    chebyshev_mapping
    logistic_mapping
    modified_lm
```

## Reservoirs

```@docs
    rand_sparse
    delay_line
    delay_line_backward
    cycle_jumps
    simple_cycle
    pseudo_svd
    chaotic_init
    low_connectivity
    double_cycle
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
    self_loop!
    add_jumps!
```
