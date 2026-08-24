# A hydrodynamic reservoir: solitons in the Korteweg-de Vries equation

[Marcucci2023](@cite) proposes Aqua-PACMANN, a reservoir computer whose
"reservoir" is a solitary wave (soliton) on shallow water: two small input
waves are launched at a fixed soliton, the resulting collision reshapes the
wave field in an input-dependent way, and a linear readout on a handful of
water-height samples recovers a Boolean logic gate. The whole system —
reservoir, input encoding, and readout — is governed by the Korteweg-de
Vries (KdV) equation, so it can be reproduced exactly as a numerical
simulation. This example builds that simulation with the same governing
equation, soliton, and encoding parameters as the paper and trains it to
reproduce their XNOR gate.

This is deliberately *not* wired through [`ReservoirComputer`](@ref) /
[`train`](@ref): those exist for reservoirs whose state is threaded
recurrently across a time series (washout, regularized fitting,
autoregressive rollout). Here each of the four truth-table rows is one
independent PDE solve, and the fit is closed-form — exactly the paper's
own `Wout = Y·pinv(X)`. Forcing that into the recurrent-reservoir API
would be more code for no benefit; see the note at the end for where
[`AbstractSciMLProblemReservoir`](@ref) *would* earn its keep.

## The KdV equation and its soliton

In normalized units, the KdV equation is
$u_t + u u_x + \beta u_{xxx} = 0$. It admits a traveling "soliton on a
pedestal": a hump of amplitude $r_1 - r_2$ sitting on a background level
$r_2$, moving at speed $v = (r_1 + 2r_2)/3$ without changing shape:

```math
u_s(x, t) = r_2 + (r_1 - r_2)\,\mathrm{sech}^2\!\left(
    \sqrt{\frac{r_1 - r_2}{12\beta}}\,(x - vt)
\right)
```

This soliton is the reservoir: a single, fixed, self-sustaining wave that
every input collides with. Following the paper, $\beta = 1/3$ and
$(r_1, r_2) = (2, 1)$, giving $v = 4/3$. The equation is solved
pseudospectrally on a periodic domain: spatial derivatives become
multiplications by $(ik)^n$ in Fourier space, and the resulting ODE is
handed to an `OrdinaryDiffEq` solver.

```@example kdv
using SciMLBase
using FFTW
using OrdinaryDiffEqTsit5
using LinearAlgebra
using Plots

N, Lx = 512, 200.0
x = collect(range(-50.0, 150.0; length = N + 1)[1:end - 1])
k = (2π / Lx) .* vcat(0:(N ÷ 2 - 1), 0, (-N ÷ 2 + 1):-1)
ik, ik3 = im .* k, (im .* k) .^ 3

function kdv_rhs!(du, u, p, t)
    uhat = fft(u)
    ux = real(ifft(p.ik .* uhat))
    uxxx = real(ifft(p.ik3 .* uhat))
    @. du = -u * ux - p.β * uxxx
    return nothing
end

β = 1 / 3
r1, r2 = 2.0, 1.0
κ = sqrt((r1 - r2) / (12β))
v_soliton = (r1 + 2r2) / 3
soliton(xx, t) = r2 + (r1 - r2) * sech(κ * (xx - v_soliton * t))^2
```

## Encoding two Boolean inputs as waves

Each input channel is a truncated, windowed wave: a `cos²` carrier at its
own wavenumber, confined to a region of width `l` by a super-Gaussian
envelope so it doesn't perturb the rest of the domain. Wavenumber picks the
channel, amplitude picks its Boolean value (`0` or `1/4`, matching the
paper). The two encoding waves plus the soliton — shifted by a delay `L`
so it starts well to the left of the encoding region — become the initial
condition:

```math
u_0(x) = \underbrace{e^{-(2x/l)^8}
    \sum_{n=1}^{2} \epsilon_n \cos^2(k_n x)}_{\text{encoded input}}
    + u_s(x + L, 0)
```

```@example kdv
L, l = 17.0, 20.0
k1, k2 = sqrt(3) / 4, 1 / 2
envelope(xx) = exp(-(2xx / l)^8)
encode(xx, ϵ1, ϵ2) = envelope(xx) * (ϵ1 * cos(k1 * xx)^2 + ϵ2 * cos(k2 * xx)^2)

build_u0(a, b) = encode.(x, 0.25a, 0.25b) .+ soliton.(x .+ L, 0.0)
```

The soliton starts at $x=-17$ moving right at $v=4/3$; a detector placed at
$x_D=50$ therefore sees it cross around $t \approx 50$. Sampling the field
there at four times bracketing that crossing — as in the paper,
$t \in \{40, 49, 51, 60\}$ — is enough state to separate all four input
combinations.

## Solving once per input and fitting the readout

Each truth-table row gets its own fresh solve from `t=0`, sampled at the
four detection times:

```@example kdv
xD_idx = argmin(abs.(x .- 50.0))
times = [40.0, 49.0, 51.0, 60.0]
p = (ik = ik, ik3 = ik3, β = β)
base_prob = ODEProblem(kdv_rhs!, build_u0(0.0, 0.0), (0.0, 60.0), p)

function response_column(a, b)
    sol = solve(remake(base_prob; u0 = build_u0(a, b)), Tsit5();
        saveat = times, reltol = 1.0e-8, abstol = 1.0e-10)
    return [sol.u[i][xD_idx] for i in eachindex(times)]
end

data = [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0]     # rows: input A, input B
X = reduce(hcat, [response_column(data[1, i], data[2, i]) for i in 1:4])

xnor_onehot(a, b) = (a == b) ? [0.0, 1.0] : [1.0, 0.0]
target = reduce(hcat, [xnor_onehot(data[1, i], data[2, i]) for i in 1:4])

Wout = target * pinv(X)
pred = Wout * X
```

## Results

```@example kdv
println("response matrix (water height at the 4 sample times, per input):")
display(X)
println("\ndet(response matrix) = ", det(X))
println("\nmax |prediction - target| = ", maximum(abs.(pred .- target)))
```

The response matrix is nonsingular — the four inputs really do land at
four linearly independent points in state space — with a determinant of
about `-0.011`, closely matching the paper's reported `-0.0115` and a good
sign that this simulation reproduces their setup faithfully. The fit
recovers the truth table to numerical precision, since it's an exact
(noiseless) linear solve of a square, invertible system.

A picture of one collision makes the mechanism concrete: the soliton
approaches from the left, passes through the encoding wave sitting near
$x=0$, and the collision leaves a wake that the detector (dashed line)
samples on its way past. Axes and ranges follow the paper's own spacetime
figure: `x` horizontal, `t` vertical, both spanning `0` to `100` (so the
solve below runs past the last detection time, $t=60$, purely to fill out
that window):

```@example kdv
u0_11 = build_u0(1.0, 1.0)
sol_11 = solve(remake(base_prob; u0 = u0_11, tspan = (0.0, 100.0)), Tsit5();
    reltol = 1.0e-8, abstol = 1.0e-10, saveat = 0:0.5:100)
U = reduce(hcat, sol_11.u)

hm = heatmap(x, sol_11.t, U'; xlabel = "x", ylabel = "t", color = :viridis,
    title = "KdV field u(x,t), inputs A=B=true", colorbar_title = "u",
    xlims = (0, 100), ylims = (0, 100))
vline!(hm, [50.0]; color = :red, linestyle = :dash, label = "detector (x=50)")
```

## When the reservoir-computer abstraction *would* help

If this KdV system needed to interoperate with the rest of
ReservoirComputing.jl — regularized fitting, washout, autoregressive
[`predict`](@ref), composition with `state_modifiers` — the right move is
what the [SciML-reservoir tutorial](@ref "Continuous-Time Reservoirs from
a `SciMLProblem`") shows: subtype [`AbstractSciMLProblemReservoir`](@ref)
and implement its `__collectstates` developer hook for your task's input
semantics (here, that the input picks the *initial condition* rather than
driving a continuous signal), then everything downstream — [`train`](@ref),
[`predict`](@ref), [`LinearReadout`](@ref) — works unchanged. For a
four-sample, closed-form fit that machinery has nothing to add, so this
example skips it.
