using Statistics, Printf

function numerical_derivative(f, x, h=1e-8)
    # Numerical derivative using central difference
    return (f(x + h) - f(x - h)) / (2h)
end

function gradient(f, x::Vector, h=1e-8)
    # Numerical gradient for multivariable functions
    n = length(x)
    grad = zeros(n)
    for i in 1:n
        x_plus = copy(x)
        x_minus = copy(x)
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2h)
    end
    return grad
end

function basic_statistics()
    # Sample data
    data = [12.5, 15.3, 18.7, 11.2, 20.1, 14.8, 16.9, 13.4, 19.2, 17.6]
    
    println("Dataset: $data")
    println("Mean: $(mean(data))")
    println("Median: $(median(data))")
    println("Standard Deviation: $(std(data))")
    println("Variance: $(var(data))")
    println("Minimum: $(minimum(data))")
    println("Maximum: $(maximum(data))")
    
    # Quartiles
    q1 = quantile(data, 0.25)
    q3 = quantile(data, 0.75)
    println("First Quartile (Q1): $q1")
    println("Third Quartile (Q3): $q3")
    println("Interquartile Range (IQR): $(q3 - q1)")
end

# Test functions
println("=== Calculus Examples ===")
f(x) = x^2 + 3x + 2
x_point = 2.0
derivative = numerical_derivative(f, x_point)
@printf "f(x) = x² + 3x + 2\n"
@printf "f'(%.1f) = %.4f\n" x_point derivative

# Multivariable function
g(x) = x[1]^2 + x[2]^2 + x[1]*x[2]
x_vec = [1.0, 2.0]
grad = gradient(g, x_vec)
println("Gradient of g(x,y) = x² + y² + xy at [1,2]: $grad")

println("\n=== Statistics Examples ===")
basic_statistics()