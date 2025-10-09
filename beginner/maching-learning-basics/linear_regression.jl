using GLM, DataFrames, Random, Statistics

function linear_regression_demo()
    Random.seed!(42)
    
    # Create sample data: house prices based on square footage
    n_houses = 50
    square_footage = rand(800:2500, n_houses)
    price = 50000 .+ 200 .* square_footage .+ randn(n_houses) .* 15000
    
    data = DataFrame(
        square_footage = square_footage,
        price = price
    )
    
    println("Housing Data Summary:")
    println("Square footage range: $(minimum(square_footage)) - $(maximum(square_footage)) sq ft")
    println("Price range: \$$(minimum(price)) - \$$(maximum(price))")
    println("Correlation: $(cor(square_footage, price))")
    
    # Fit linear regression model
    model = lm(@formula(price ~ square_footage), data)
    
    println("\n=== Linear Regression Results ===")
    println(model)
    
    # Make predictions
    new_sizes = DataFrame(square_footage = [1000, 1500, 2000, 2500])
    predictions = predict(model, new_sizes)
    
    println("\nPredictions for new houses:")
    for (size, pred) in zip(new_sizes.square_footage, predictions)
        @printf "%d sq ft -> \$%.2f\n" size pred
    end
    
    # Model evaluation
    y_pred = predict(model, data)
    r_squared = r2(model)
    mse = mean((data.price - y_pred).^2)
    
    println("\nModel Evaluation:")
    println("R-squared: $(r_squared)")
    println("Mean Squared Error: $(mse)")
    
    return model, data
end

function multiple_regression()
    Random.seed!(42)
    
    # Multiple regression with more features
    n_samples = 100
    data = DataFrame(
        study_hours = rand(1:20, n_samples),
        attendance = rand(70:100, n_samples),
        previous_score = rand(50:90, n_samples),
        final_score = rand(50:95, n_samples)
    )
    
    # Fit multiple regression
    model = lm(@formula(final_score ~ study_hours + attendance + previous_score), data)
    
    println("\n=== Multiple Linear Regression ===")
    println("Final Score ~ Study Hours + Attendance + Previous Score")
    println(model)
    
    return model
end

# Run the demos
linear_model, housing_data = linear_regression_demo()
multiple_model = multiple_regression()