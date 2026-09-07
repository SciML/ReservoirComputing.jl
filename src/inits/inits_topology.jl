function __topology_spec(func::Symbol)
    func === :delay_line! && return (:delay, :shift, 1)
    func === :backward_connection! && return (:fb, :shift, 1)
    func === :simple_cycle! && return (:cycle, nothing, nothing)
    func === :reverse_simple_cycle! && return (:second_cycle, nothing, nothing)
    func === :self_loop! && return (:selfloop, nothing, nothing)
    func === :add_jumps! && return (:jump, :jump_size, 3)
    func === :permute_matrix! && return (:permute, :permutation_matrix, nothing)
    return throw(ArgumentError("@topology unknown building block `$func`"))
end

function __topology_kw(prefix::Symbol, param::Symbol)
    param === :permutation_matrix && return param
    startswith(String(param), String(prefix)) && return param
    return Symbol(prefix, :_, param)
end

function __topology_weight_kw(prefix::Symbol)
    prefix === :weight && return :weight
    return Symbol(prefix, :_weight)
end

function __topology_kws(ex)
    kws = Dict{Symbol, Any}()
    Meta.isexpr(ex, :call) || return kws
    for arg in ex.args[2:end]
        params = Meta.isexpr(arg, :parameters) ? arg.args : (arg,)
        for p in params
            Meta.isexpr(p, :kw) || throw(
                ArgumentError("@topology building blocks take keywords, got $p")
            )
            kws[p.args[1]] = p.args[2]
        end
    end
    return kws
end

function __topology_entry(stmt)
    prefix = nothing
    if Meta.isexpr(stmt, :(=), 2) && stmt.args[1] isa Symbol
        prefix, stmt = stmt.args
    end
    func = Meta.isexpr(stmt, :call) ? stmt.args[1] : stmt
    func isa Symbol && endswith(String(func), "!") || throw(
        ArgumentError("@topology expected a bang call, got $stmt")
    )
    return prefix, func, __topology_kws(stmt)
end

function __topology_typed_kw(name, typ, default)
    return Expr(:kw, Expr(:(::), name, typ), default)
end

function __topology_expand(name::Symbol, body)
    stmts = Meta.isexpr(body, :block) ?
        filter(s -> !(s isa LineNumberNode), body.args) : Any[body]
    isempty(stmts) && throw(
        ArgumentError("@topology requires at least one building block")
    )

    entries = map(__topology_entry, stmts)
    n_signs = count(e -> e[2] !== :permute_matrix!, entries)
    used = Set{Symbol}()
    function claim(kw)
        kw in used && throw(ArgumentError("@topology duplicate keyword `$kw`"))
        push!(used, kw)
        return kw
    end

    sig = Any[]
    sign_kws = Any[]
    calls = Expr[]
    for (alias, func, kws) in entries
        spec_prefix, extra, extra_default = __topology_spec(func)
        weight_prefix = alias === nothing ? spec_prefix : alias
        weighted = func !== :permute_matrix!
        allowed = extra === nothing ? (:weight,) : (:weight, extra)
        leftover = setdiff(keys(kws), allowed)
        isempty(leftover) || throw(
            ArgumentError("@topology `$func` got unsupported keyword $(join(leftover, ", "))")
        )

        args = Any[:rng, :reservoir_matrix]
        if weighted
            wkw = claim(__topology_weight_kw(weight_prefix))
            wdef = get(kws, :weight, nothing)
            wdef = wdef === nothing ? :(T(0.1f0)) :
                wdef isa Number ? :(T($wdef)) : wdef
            push!(sig, __topology_typed_kw(wkw, :(Union{Number, AbstractVector}), wdef))
            push!(args, :(T.($wkw)))
        end
        if extra !== nothing
            ekw = claim(
                alias === nothing ? __topology_kw(spec_prefix, extra) : extra
            )
            etype = extra === :permutation_matrix ?
                :(Union{Nothing, AbstractMatrix}) : :Integer
            push!(sig, __topology_typed_kw(ekw, etype, get(kws, extra, extra_default)))
            push!(args, ekw)
        end
        bang = getfield(@__MODULE__, func)
        if weighted
            if n_signs == 1
                splat = :kwargs
            else
                splat = claim(Symbol(spec_prefix, :_kwargs))
                push!(sign_kws, __topology_typed_kw(splat, :NamedTuple, :(NamedTuple())))
            end
            push!(calls, Expr(:call, bang, Expr(:parameters, :($splat...)), args...))
        else
            push!(calls, Expr(:call, bang, args...))
        end
    end
    push!(
        sig,
        __topology_typed_kw(:radius, :(Union{AbstractFloat, Nothing}), :nothing),
        __topology_typed_kw(:return_sparse, :Bool, :false),
    )
    if n_signs == 1
        push!(sig, :(kwargs...))
    else
        append!(sig, sign_kws)
    end

    primary = Expr(
        :function,
        Expr(
            :where,
            Expr(
                :call, name, Expr(:parameters, sig...),
                :(rng::$AbstractRNG), :(::Type{T}), :(dims::Integer...),
            ),
            :(T <: Number),
        ),
        quote
            $throw_sparse_error(return_sparse)
            $check_res_size(dims...)
            reservoir_matrix = $DeviceAgnostic.zeros(rng, T, dims...)
            $(calls...)
            $scale_radius!(reservoir_matrix, radius)
            return $return_init_as(Val(return_sparse), reservoir_matrix)
        end,
    )
    return quote
        Base.@__doc__ $primary
        function $name(dims::Integer...; kwargs...)
            return $name($Utils.default_rng(), Float32, dims...; kwargs...)
        end
        function $name(rng::$AbstractRNG, dims::Integer...; kwargs...)
            return $name(rng, Float32, dims...; kwargs...)
        end
        function $name(::Type{T}, dims::Integer...; kwargs...) where {T <: Number}
            return $name($Utils.default_rng(), T, dims...; kwargs...)
        end
        function $name(rng::$AbstractRNG; kwargs...)
            return $PartialFunction.Partial{Nothing}($name, rng, kwargs)
        end
        function $name(::Type{T}; kwargs...) where {T <: Number}
            return $PartialFunction.Partial{T}($name, nothing, kwargs)
        end
        function $name(
                rng::$AbstractRNG, ::Type{T}; kwargs...
            ) where {T <: Number}
            return $PartialFunction.Partial{T}($name, rng, kwargs)
        end
        function $name(; kwargs...)
            return $PartialFunction.Partial{Nothing}($name, nothing, kwargs)
        end
    end
end

@doc raw"""
    @topology name begin
        block!
        alias = block!(; extra=default)
    end

Create a reservoir initializer from building blocks such as
[`self_loop!`](@ref), [`simple_cycle!`](@ref), and [`delay_line!`](@ref).

Each block has a `{prefix}_weight` keyword (default `0.1`). Extra arguments
become keywords (`delay_shift`, `fb_shift`, `jump_size`). `radius` and
`return_sparse` are always available. With more than one block, pass sign
patterns as `{prefix}_kwargs`, for example
`cycle_kwargs=(; signs=RandomSigns())`.

`alias = block!` names the weight (`forward = delay_line!` → `forward_weight`,
`weight = self_loop!` → `weight`). Sign kwargs stay on the block
(`delay_kwargs`). An aliased extra keeps the bang name (`shift`); otherwise
it is `{prefix}_{arg}` (`delay_shift`). Defaults: `self_loop!` → `selfloop`,
`simple_cycle!` → `cycle`, `delay_line!` → `delay`,
`backward_connection!` → `fb`, `add_jumps!` → `jump`,
`reverse_simple_cycle!` → `second_cycle`, `permute_matrix!` → `permute`.

```jldoctest
julia> @topology my_cycle begin
           self_loop!
           simple_cycle!
       end;

julia> my_cycle(5, 5)
5×5 Matrix{Float32}:
 0.1  0.0  0.0  0.0  0.1
 0.1  0.1  0.0  0.0  0.0
 0.0  0.1  0.1  0.0  0.0
 0.0  0.0  0.1  0.1  0.0
 0.0  0.0  0.0  0.1  0.1
```
"""
macro topology(name, body)
    name isa Symbol || throw(
        ArgumentError("@topology first argument must be an identifier, got $name")
    )
    return esc(__topology_expand(name, body))
end
