function leaky_fixed_rnn(activation, alpha, W, W_in, x, y)
    return (1-alpha).*x + alpha*activation.((W*x)+(W_in*y))
end

function double_leaky_fixed_rnn(alpha, f_activation, s_activation, f_lambda, s_lambda, W, W_in, x, y)
    return (1-alpha).*x + f_lambda*f_activation.((W*x)+(W_in*y)) + s_lambda*s_activation.((W*x)+(W_in*y))
end
