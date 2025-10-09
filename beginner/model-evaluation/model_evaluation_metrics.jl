using DataFrames, Random, Statistics, Plots

function classification_metrics_demo()
    Random.seed!(42)
    
    # Generate sample classification results
    n_samples = 100
    y_true = rand([0, 1], n_samples)
    y_pred = copy(y_true)
    
    # Introduce some errors
    error_indices = rand(1:n_samples, 15)
    for idx in error_indices
        y_pred[idx] = 1 - y_true[idx]  # Flip prediction
    end
    
    y_proba = zeros(n_samples)
    for i in 1:n_samples
        if y_pred[i] == y_true[i]
            y_proba[i] = rand(0.6:0.01:0.95)  # High probability for correct
        else
            y_proba[i] = rand(0.4:0.01:0.6)   # Medium probability for errors
        end
    end
    
    # Calculate metrics
    accuracy = mean(y_pred .== y_true)
    
    # Confusion matrix
    tp = sum((y_pred .== 1) .& (y_true .== 1))
    fp = sum((y_pred .== 1) .& (y_true .== 0))
    tn = sum((y_pred .== 0) .& (y_true .== 0))
    fn = sum((y_pred .== 0) .& (y_true .== 1))
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # ROC curve calculations (simplified)
    thresholds = 0.0:0.1:1.0
    tpr = Float64[]
    fpr = Float64[]
    
    for threshold in thresholds
        pred_thresh = y_proba .>= threshold
        tp_rate = sum((pred_thresh .== 1) .& (y_true .== 1)) / sum(y_true .== 1)
        fp_rate = sum((pred_thresh .== 1) .& (y_true .== 0)) / sum(y_true .== 0)
        push!(tpr, tp_rate)
        push!(fpr, fp_rate)
    end
    
    # AUC (simplified)
    auc = trapz(fpr, tpr)
    
    println("=== Classification Metrics ===")
    println("Accuracy:  $(round(accuracy, digits=4))")
    println("Precision: $(round(precision, digits=4))")
    println("Recall:    $(round(recall, digits=4))")
    println("F1-Score:  $(round(f1, digits=4))")
    println("AUC:       $(round(auc, digits=4))")
    
    println("\nConfusion Matrix:")
    println("True Positives:  $tp")
    println("False Positives: $fp")
    println("True Negatives:  $tn")
    println("False Negatives: $fn")
    
    # Plot ROC curve
    plot(fpr, tpr, linewidth=2, label="ROC Curve (AUC = $(round(auc, digits=3)))")
    plot!([0, 1], [0, 1], linestyle=:dash, color=:gray, label="Random Classifier")
    xlabel!("False Positive Rate")
    ylabel!("True Positive Rate")
    title!("ROC Curve")
    
    return (accuracy=accuracy, precision=precision, recall=recall, f1=f1, auc=auc)
end

function regression_metrics_demo()
    Random.seed!(42)
    
    # Generate sample regression results
    n_samples = 50
    x = 1:n_samples
    y_true = 10 .+ 0.5 .* x .+ randn(n_samples) .* 2
    y_pred = 10 .+ 0.48 .* x .+ randn(n_samples) .* 2.5  # Slightly different model
    
    # Calculate regression metrics
    mse = mean((y_true - y_pred).^2)
    rmse = sqrt(mse)
    mae = mean(abs.(y_true - y_pred))
    
    # R-squared
    ss_res = sum((y_true - y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Mean Absolute Percentage Error
    mape = mean(abs.((y_true - y_pred) ./ y_true)) * 100
    
    println("\n=== Regression Metrics ===")
    println("Mean Squared Error (MSE):  $(round(mse, digits=4))")
    println("Root Mean Squared Error (RMSE): $(round(rmse, digits=4))")
    println("Mean Absolute Error (MAE): $(round(mae, digits=4))")
    println("R-squared: $(round(r_squared, digits=4))")
    println("Mean Absolute Percentage Error: $(round(mape, digits=2))%")
    
    # Plot results
    scatter(x, y_true, label="True Values", alpha=0.7)
    plot!(x, y_pred, linewidth=2, label="Predictions", color=:red)
    xlabel!("Sample Index")
    ylabel!("Target Value")
    title!("Regression: True vs Predicted")
    
    # Residual plot
    residuals = y_true - y_pred
    scatter(y_pred, residuals, alpha=0.7, color=:green)
    plot!([minimum(y_pred), maximum(y_pred)], [0, 0], linestyle=:dash, color=:gray, label="Zero Line")
    xlabel!("Predicted Values")
    ylabel!("Residuals")
    title!("Residual Plot")
    
    return (mse=mse, rmse=rmse, mae=mae, r_squared=r_squared, mape=mape)
end

function train_validation_curves()
    Random.seed!(42)
    
    # Demonstrate learning curves
    n_samples_range = 20:20:200
    train_scores = Float64[]
    val_scores = Float64[]
    
    for n_samples in n_samples_range
        # Generate data
        X_train = randn(n_samples, 2)
        y_train = 2 .* X_train[:,1] .- X_train[:,2] .+ randn(n_samples) * 0.5
        
        # Simple model (linear regression coefficients)
        if n_samples >= 2
            # Manual linear regression
            X_design = hcat(ones(n_samples), X_train)
            coefficients = X_design \ y_train
            
            # Training score (R²)
            y_pred_train = X_design * coefficients
            train_r2 = 1 - sum((y_train - y_pred_train).^2) / sum((y_train .- mean(y_train)).^2)
            push!(train_scores, train_r2)
            
            # Validation score (on new data)
            X_val = randn(100, 2)
            y_val = 2 .* X_val[:,1] .- X_val[:,2] .+ randn(100) * 0.5
            X_val_design = hcat(ones(100), X_val)
            y_pred_val = X_val_design * coefficients
            val_r2 = 1 - sum((y_val - y_pred_val).^2) / sum((y_val .- mean(y_val)).^2)
            push!(val_scores, val_r2)
        else
            push!(train_scores, 0.0)
            push!(val_scores, 0.0)
        end
    end
    
    # Plot learning curves
    plot(n_samples_range, train_scores, linewidth=2, label="Training Score")
    plot!(n_samples_range, val_scores, linewidth=2, label="Validation Score")
    xlabel!("Training Set Size")
    ylabel!("R² Score")
    title!("Learning Curves")
    
    println("\n=== Learning Curve Analysis ===")
    println("Final training score: $(round(train_scores[end], digits=4))")
    println("Final validation score: $(round(val_scores[end], digits=4))")
    println("Gap: $(round(train_scores[end] - val_scores[end], digits=4))")
    
    return (train_scores=train_scores, val_scores=val_scores)
end

# Numerical integration for AUC
function trapz(x, y)
    n = length(x)
    integral = 0.0
    for i in 1:n-1
        integral += (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2
    end
    return integral
end

# Run evaluation demos
classification_metrics = classification_metrics_demo()
regression_metrics = regression_metrics_demo()
learning_curves = train_validation_curves()