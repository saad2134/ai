using LinearAlgebra

function vector_operations()
    # Vector operations
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    
    println("Vector 1: $v1")
    println("Vector 2: $v2")
    
    # Addition
    addition = v1 + v2
    println("Addition: $addition")
    
    # Dot product
    dot_product = dot(v1, v2)
    println("Dot product: $dot_product")
    
    # Cross product
    cross_product = cross(v1, v2)
    println("Cross product: $cross_product")
    
    # Magnitude
    magnitude_v1 = norm(v1)
    println("Magnitude of v1: $magnitude_v1")
    
    # Matrix operations
    A = [1 2; 3 4]
    B = [5 6; 7 8]
    
    println("\nMatrix A:")
    println(A)
    println("Matrix B:")
    println(B)
    
    # Matrix multiplication
    C = A * B
    println("Matrix multiplication A * B:")
    println(C)
    
    # Determinant
    det_A = det(A)
    println("Determinant of A: $det_A")
    
    # Eigenvalues
    eigenvals = eigen(A).values
    println("Eigenvalues of A: $eigenvals")
end

function manual_dot_product(v1, v2)
    # Manual dot product implementation
    if length(v1) != length(v2)
        error("Vectors must have same length")
    end
    result = 0.0
    for i in 1:length(v1)
        result += v1[i] * v2[i]
    end
    return result
end

# Run the functions
vector_operations()

v1 = [1.0, 2.0, 3.0]
v2 = [4.0, 5.0, 6.0]
println("\nManual dot product: $(manual_dot_product(v1, v2))")