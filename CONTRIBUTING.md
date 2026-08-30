# Contributing

ReservoirComputing.jl follows the SciML contribution process. Please read:

- [SciML CONTRIBUTING](https://github.com/SciML/.github/blob/master/CONTRIBUTING.md)
- [SciML Style](https://github.com/SciML/SciMLStyle)
- [ColPrac](https://github.com/SciML/ColPrac)

The notes below are extra conventions this library enforces on top of those.

## Commits

This repository uses [conventional commits](https://www.conventionalcommits.org/)
(`docs:`, `fix:`, `feat:`, `chore:`, ...).

Keep the subject short. Put what changed in the body. Do not add LLM
`Co-Authored-By` trailers.

## Naming

Use domain names, not single letters: `current_state` not `s`;
`reservoir_matrix` / `input_matrix` / `weight` not `A` / `W`.

Loop indices `i, j, k` and paper equation variables are fine when translating
a formula line-for-line.

## Models and cells

Models are three fields only: `(reservoir, state_modifiers, readout)`.
Do not add config fields on the model wrapper.

Cells use `(in_dims, out_dims)::Pair` constructors and kwargs. Do not invent
a new cell if `ESNCell(...; activation=identity)` already does the job.

## Types, errors, numerics

- Integers: `IntegerType` (never bare `Int` for stored dims).
- Bool kwargs: `BoolType` / `static`.
- User errors: `ArgumentError` / `DimensionMismatch`, not `@assert`.
- Infer eltype from data or inits. Tests default to `Float32`.

## Docs, comments, format

Docstrings are user-facing (`@doc raw"""..."""`). Do not document dispatch,
package extensions, or helper internals. One canonical paper cite is enough.

Comments explain *why*. Do not restate the next line.

Format with [Runic](https://github.com/fredrikekre/Runic.jl). It is a global
tool, not a `Project.toml` dependency. There is no `.JuliaFormatter.toml`.

```bash
julia -e 'using Runic; exit(Int(Runic.main(["src", "ext", "test"])))'
```

## Tests

Tests are grouped with SciMLTesting (`GROUP=Core`, `GROUP=Layers`,
`GROUP=Models`, ...). Seed RNGs with `MersenneTwister(42)` or the seeds
already in the file.

When in doubt, match the nearest sibling file.
