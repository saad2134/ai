using MLJ, DataFrames, Random, Statistics

function comprehensive_cross_validation()
    Random.seed!(42)
    
    # Create classification dataset
    n_samples = 200
    X, y = make_blobs(n_samples, 2, 3; centers=3, cluster_std=1.5, random_state=42)
    
    data = DataFrame(X1=X[:,1], X2=X[:,2], class=y)
    
    # Prepare for MLJ
    y_cat = categorical(y)
    
    # Define models
    models = Dict(
        "Decision Tree" => @load DecisionTreeClassifier pkg=DecisionTree,
        "Random Forest" => @load RandomForestClassifier pkg=DecisionTree,
        "KNN" => @load KNNClassifier pkg=NearestNeighborModels
    )
    
    println("=== 5-Fold Cross-Validation Results ===")
    
    results = DataFrame()
    for (name, model_type) in models
        model = model_type()
        
        # Perform 5-fold cross-validation
        cv = CV(nfolds=5, shuffle=true, rng=42)
        evaluation = evaluate(model, X, y_cat, 
                            resampling=cv,
                            measure=[accuracy, precision, recall, f1score],
                            operation=predict_mode)
        
        # Store results
        result_row = DataFrame(
            Model = name,
            Accuracy = round(mean(evaluation.measurement[1]), digits=4),
            Precision = round(mean(evaluation.measurement[2]), digits=4),
            Recall = round(mean(evaluation.measurement[3]), digits=4),
            F1_Score = round(mean(evaluation.measurement[4]), digits=4)
        )
        
        results = vcat(results, result_row)
        
        println("$name:")
        println("  Accuracy:  $(result_row.Accuracy[1])")
        println("  Precision: $(result_row.Precision[1])")
        println("  Recall:    $(result_row.Recall[1])")
        println("  F1-Score:  $(result_row.F1_Score[1])")
        println()
    end
    
    # Find best model
    best_idx = argmax(results.F1_Score)
    best_model = results.Model[best_idx]
    println("Best model: $best_model (F1-Score: $(results.F1_Score[best_idx]))")
    
    return results
end

function regression_cross_validation()
    Random.seed!(42)
    
    # Create regression dataset
    n_samples = 150
    X = randn(n_samples, 3)
    y = 2.5 .+ 1.8 * X[:,1] .- 0.9 * X[:,2] .+ 0.5 * X[:,3] .+ randn(n_samples) * 0.5
    
    # Define regression models
    models = Dict(
        "Linear Regression" => @load LinearRegressor pkg=MLJLinearModels,
        "Decision Tree" => @load DecisionTreeRegressor pkg=DecisionTree,
        "Random Forest" => @load RandomForestRegressor pkg=DecisionTree
    )
    
    println("=== Regression Cross-Validation Results ===")
    
    results = DataFrame()
    for (name, model_type) in models
        model = model_type()
        
        # Perform 5-fold cross-validation
        cv = CV(nfolds=5, shuffle=true, rng=42)
        evaluation = evaluate(model, X, y,
                            resampling=cv,
                            measure=[rms, mae, rsq],
                            operation=predict)
        
        # Store results
        result_row = DataFrame(
            Model = name,
            RMSE = round(mean(evaluation.measurement[1]), digits=4),
            MAE = round(mean(evaluation.measurement[2]), digits=4),
            R_Squared = round(mean(evaluation.measurement[3]), digits=4)
        )
        
        results = vcat(results, result_row)
        
        println("$name:")
        println("  RMSE:      $(result_row.RMSE[1])")
        println("  MAE:       $(result_row.MAE[1])")
        println("  R-Squared: $(result_row.R_Squared[1])")
        println()
    end
    
    return results
end

# Helper function to create sample data
function make_blobs(n_samples, n_features, n_centers; centers=nothing, cluster_std=1.0, random_state=42)
    Random.seed!(random_state)
    
    if centers === nothing
        centers = randn(n_centers, n_features) * 2
    end
    
    X = zeros(n_samples, n_features)
    y = zeros(Int, n_samples)
    
    samples_per_center = n_samples ÷ n_centers
    sample_count = 0
    
    for center_idx in 1:n_centers
        n_center_samples = (center_idx == n_centers) ? (n_samples - sample_count) : samples_per_center
        
        for i in 1:n_center_samples
            X[sample_count + i, :] = centers[center_idx, :] .+ randn(n_features) .* cluster_std
            y[sample_count + i] = center_idx
        end
        
        sample_count += n_center_samples
    end
    
    # Shuffle
    shuffle_idx = shuffle(1:n_samples)
    return X[shuffle_idx, :], y[shuffle_idx]
end

# Run cross-validation demos
classification_results = comprehensive_cross_validation()
regression_results = regression_cross_validation()