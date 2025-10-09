using MultivariateStats, DataFrames, Random, Plots, LinearAlgebra

function pca_iris_visualization()
    # Apply PCA to Iris dataset for 2D visualization
    iris = load_iris()
    X = Matrix(iris.features)
    y = iris.targets
    
    println("Original data shape: $(size(X))")
    println("Features: Sepal Length, Sepal Width, Petal Length, Petal Width")
    
    # Apply PCA
    M = fit(PCA, X'; maxoutdim=2)
    X_proj = transform(M, X')'
    
    println("\n=== PCA Results ===")
    println("Principal components calculated")
    println("Projected data shape: $(size(X_proj))")
    
    # Explained variance
    principal_ratio = principalratio(M)
    println("Variance explained by PC1: $(round(principal_ratio[1] * 100, digits=1))%")
    println("Variance explained by PC2: $(round(principal_ratio[2] * 100, digits=1))%")
    println("Total variance explained: $(round(sum(principal_ratio) * 100, digits=1))%")
    
    # Create DataFrame for plotting
    pca_df = DataFrame(
        PC1 = X_proj[:, 1],
        PC2 = X_proj[:, 2],
        Species = y
    )
    
    # Plot PCA results
    scatter(pca_df.PC1, pca_df.PC2, group=pca_df.Species,
            marker=:auto, alpha=0.7, legend=:topright,
            xlabel="Principal Component 1",
            ylabel="Principal Component 2",
            title="PCA: Iris Dataset Visualization")
    
    return M, pca_df
end

function pca_feature_analysis()
    Random.seed!(42)
    
    # Create high-dimensional dataset
    n_samples = 200
    n_features = 10
    
    # Generate correlated features
    X = randn(n_samples, n_features)
    # Create correlation between features
    for i in 2:n_features
        X[:, i] += 0.5 * X[:, i-1]
    end
    
    println("High-dimensional Dataset:")
    println("Shape: $n_samples samples × $n_features features")
    
    # Apply PCA
    M = fit(PCA, X'; maxoutdim=4)
    X_proj = transform(M, X')'
    
    # Analyze components
    println("\n=== PCA Feature Analysis ===")
    println("Principal Components:")
    
    # Get loadings (component coefficients)
    loadings = projection(M)
    
    for i in 1:min(3, size(loadings, 2))
        println("PC$i loadings: $(round.(loadings[:, i], digits=3))")
    end
    
    # Scree plot
    principal_ratio = principalratio(M)
    plot(1:length(principal_ratio), cumsum(principal_ratio), 
         marker=:o, linewidth=2, legend=false,
         xlabel="Number of Components",
         ylabel="Cumulative Explained Variance",
         title="Scree Plot")
    
    # Find number of components for 95% variance
    n_components_95 = findfirst(x -> x >= 0.95, cumsum(principal_ratio))
    println("Components needed for 95% variance: $n_components_95")
    
    return M, X_proj
end

function pca_compression_demo()
    # Demonstrate PCA for data compression
    Random.seed!(42)
    
    # Create sample image-like data (flattened 8x8 images)
    n_images = 100
    original_dim = 64
    X_images = randn(n_images, original_dim)
    
    println("Original data dimension: $original_dim")
    
    # Apply PCA with different numbers of components
    components_range = [2, 5, 10, 20, 32]
    compression_ratios = Float64[]
    explained_variances = Float64[]
    
    for n_comp in components_range
        M = fit(PCA, X_images'; maxoutdim=n_comp)
        X_compressed = transform(M, X_images')
        X_reconstructed = reconstruct(M, X_compressed)'
        
        # Calculate compression ratio
        original_size = n_images * original_dim
        compressed_size = n_images * n_comp + n_comp * original_dim  # Data + components
        ratio = original_size / compressed_size
        push!(compression_ratios, ratio)
        
        # Calculate reconstruction error
        variance_explained = sum(principalratio(M))
        push!(explained_variances, variance_explained)
        
        println("Components: $n_comp, Compression: $(round(ratio, digits=1)):1, Variance: $(round(variance_explained*100, digits=1))%")
    end
    
    # Plot trade-off
    plot(components_range, compression_ratios, marker=:o, linewidth=2,
         label="Compression Ratio", xlabel="Number of Components",
         ylabel="Compression Ratio", title="PCA Compression Trade-off")
    
    plot!(twinx(), components_range, explained_variances, 
          marker=:s, linewidth=2, color=:red, label="Variance Explained",
          ylabel="Explained Variance Ratio", legend=:right)
end

# Run PCA demos
pca_model, iris_pca_df = pca_iris_visualization()
pca_high_dim, high_dim_proj = pca_feature_analysis()
pca_compression_demo()