using MLJ, DataFrames, Random

function train_test_demo()
    Random.seed!(42)
    
    # Create sample housing dataset
    n_samples = 100
    square_footage = rand(800:2500, n_samples)
    bedrooms = rand(1:4, n_samples)
    price = 50000 .+ 200 .* square_footage .+ 15000 .* bedrooms .+ randn(n_samples) .* 10000
    
    data = DataFrame(
        square_footage = square_footage,
        bedrooms = bedrooms,
        price = price
    )
    
    println("Dataset Info:")
    println("Size: $(n_samples) samples")
    println("First 5 rows:")
    show(first(data, 5), allcols=true)
    println("\n")
    
    # Split into features and target
    X = select(data, [:square_footage, :bedrooms])
    y = data.price
    
    # Train-test split (70-30)
    train_idx, test_idx = partition(eachindex(y), 0.7, shuffle=true, rng=42)
    
    X_train = X[train_idx, :]
    X_test = X[test_idx, :]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    println("Training set size: $(size(X_train, 1))")
    println("Test set size: $(size(X_test, 1))")
    
    return X_train, X_test, y_train, y_test
end

function ml_types_demo()
    println("=== Machine Learning Types ===")
    println("1. Supervised Learning:")
    println("   - Uses labeled data (input-output pairs)")
    println("   - Examples: Regression, Classification")
    println("   - Goal: Predict outputs for new inputs")
    
    println("\n2. Unsupervised Learning:")
    println("   - Uses unlabeled data")
    println("   - Examples: Clustering, Dimensionality Reduction")
    println("   - Goal: Discover patterns and structure")
    
    println("\n3. Reinforcement Learning:")
    println("   - Learns through interaction with environment")
    println("   - Examples: Game playing, Robotics")
    println("   - Goal: Learn optimal actions through rewards")
    
    println("\nKey Concepts:")
    println("- Training Data: Used to train the model")
    println("- Testing Data: Used to evaluate model performance")
    println("- Overfitting: Model learns training data too well")
    println("- Underfitting: Model fails to learn patterns")
end

# Run demos
X_train, X_test, y_train, y_test = train_test_demo()
ml_types_demo()