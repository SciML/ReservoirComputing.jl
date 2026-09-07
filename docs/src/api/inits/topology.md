# @topology

Reservoir topologies such as [`selfloop_cycle`](@ref) are stacks of bang
building blocks. [`@topology`](@ref) writes that stack as an initializer.

```@example topology
using ReservoirComputing

@topology my_cycle begin
    self_loop!
    simple_cycle!
end

my_cycle(5, 5)
```

The result is a normal initializer, so it can be passed to `init_reservoir`:

```@example topology
ESN(1, 50, 1; init_reservoir = my_cycle(; cycle_weight = 0.2))
```

`alias = block!` names that block's weight. Sign kwargs stay on the original
block (`delay_kwargs`):

```@example topology
@topology my_forward begin
    self_loop!
    forward = delay_line!(shift = 2)
end

my_forward(5, 5; forward_weight = 0.5)
```

```@docs
    @topology
```
