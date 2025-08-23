function predict(rc, steps::Int, ps, st; initialdata=nothing)
    if initialdata == nothing
        initialdata = rand(Float32, 3)
    end
    output = zeros(size(initialdata, 1), steps)
    for step in 1:steps
        initialdata, st = apply(rc, initialdata, ps, st)
        output[:, step] = initialdata
    end
    return output, st
end
