# Initializer sign patterns

Sign patterns modify the signs of weights produced by compatible input and reservoir
initializers. Passing `signs = nothing` leaves the generated signs unchanged.

!!! warning "Deprecated keyword API"
    The `sampling_type` keyword and its sampler-specific forwarded keywords are
    deprecated. Replace them with a sign-pattern object passed as `signs`. For example,
    replace `sampling_type = :bernoulli_sample!, positive_prob = 0.25` with
    `signs = RandomSigns(0.25)`.

    `minimal_init` now also defaults to `signs = nothing`; use `RandomSigns()` to retain
    its former default Bernoulli sign flipping explicitly.

```@docs
AbstractSignPattern
RandomSigns
RegularSigns
IrrationalDigitSigns
```
