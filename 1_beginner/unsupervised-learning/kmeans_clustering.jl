using Clustering, DataFrames, Random, Statistics, Plots

function kmeans_customer_segmentation()
    Random.seed!(42)
    
    # Create customer data for mall segmentation
    n_customers = 300
    data = DataFrame(
        age = rand(18:70, n_customers),
        annual_income = rand(15000:150000, n_customers),
        spending_score = rand(1:100, n_customers)
    )
    
    println("Customer Data Summary:")
    println("Age range: $(minimum(data.age)) - $(maximum(data.age))")
    println("Income range: \$$(minimum(data.annual_income)) - \$$(maximum(data.annual_income))")
    println("Spending score range: $(minimum(data.spending_score)) - $(maximum(data.spending_score))")
    
    # Prepare data for clustering (normalize)
    X = Matrix(select(data, [:age, :annual_income, :spending_score]))
    X_normalized = (X .- mean(X, dims=1)) ./ std(X, dims=1)
    
    # Find optimal number of clusters using elbow method
    k_range = 2:8
    wcss = Float64[]  # Within-cluster sum of squares
    
    for k in k_range
        result = kmeans(X_normalized', k)
        push!(wcss, result.totalcost)
    end
    
    # Plot elbow curve
    plot(k_range, wcss, marker=:o, linewidth=2, 
         title="Elbow Method for Optimal k",
         xlabel="Number of Clusters",
         ylabel="Within-Cluster Sum of Squares",
         legend=false)
    
    # Choose k=4 based on elbow method
    k_optimal = 4
    result = kmeans(X_normalized', k_optimal)
    
    # Add clusters to dataframe
    data.cluster = assignments(result)
    
    println("\n=== K-Means Clustering Results (k=$k_optimal) ===")
    println("Cluster sizes: $(counts(result))")
    
    # Analyze clusters
    cluster_summary = combine(groupby(data, :cluster), 
        :age => mean => :avg_age,
        :annual_income => mean => :avg_income,
        :spending_score => mean => :avg_spending_score,
        nrow => :count
    )
    
    println("\nCluster Profiles:")
    show(cluster_summary, allcols=true)
    println()
    
    # Visualize clusters (2D projection)
    scatter(data.annual_income, data.spending_score, 
            group=data.cluster, marker=:auto, alpha=0.7,
            xlabel="Annual Income", ylabel="Spending Score",
            title="Customer Segmentation",
            legend=:topright)
    
    return result, data
end

function kmeans_iris()
    # Apply K-Means to Iris dataset
    iris = load_iris()
    X = Matrix(iris.features)
    
    # Normalize data
    X_normalized = (X .- mean(X, dims=1)) ./ std(X, dims=1)
    
    # Apply K-Means
    result = kmeans(X_normalized', 3)
    
    # Compare with true labels
    true_labels = iris.targets
    cluster_labels = assignments(result)
    
    # Calculate accuracy (adjusted for label permutation)
    accuracy = clustering_accuracy(true_labels, cluster_labels)
    
    println("\n=== K-Means on Iris Dataset ===")
    println("True classes: $(unique(true_labels))")
    println("Found clusters: $(unique(cluster_labels))")
    println("Clustering accuracy: $(round(accuracy, digits=3))")
    
    return result
end

function clustering_accuracy(true_labels, pred_labels)
    # Simple clustering accuracy (needs improvement for proper evaluation)
    # This is a simplified version - in practice, use adjusted rand index
    correct = 0
    for i in 1:length(true_labels)
        for j in i+1:length(true_labels)
            if (true_labels[i] == true_labels[j]) == (pred_labels[i] == pred_labels[j])
                correct += 1
            end
        end
    end
    total_pairs = length(true_labels) * (length(true_labels) - 1) ÷ 2
    return correct / total_pairs
end

# Load Iris dataset
function load_iris()
    # Simplified Iris dataset loader
    # In practice, use RDatasets or similar
    features = [5.1 3.5 1.4 0.2; 4.9 3.0 1.4 0.2; 4.7 3.2 1.3 0.2;
                7.0 3.2 4.7 1.4; 6.4 3.2 4.5 1.5; 6.9 3.1 4.9 1.5;
                6.3 3.3 6.0 2.5; 5.8 2.7 5.1 1.9; 7.1 3.0 5.9 2.1]
    targets = ["setosa", "setosa", "setosa", "versicolor", "versicolor", "versicolor", 
               "virginica", "virginica", "virginica"]
    return (features=features, targets=targets)
end

# Run K-Means demos
kmeans_result, customer_data = kmeans_customer_segmentation()
iris_result = kmeans_iris()