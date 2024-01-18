function check_radius(matrix, target_radius; tolerance=1e-5)
    eigenvalues = eigvals(matrix)
    spectral_radius = maximum(abs.(eigenvalues))
    return isapprox(spectral_radius, target_radius, atol=tolerance)
end