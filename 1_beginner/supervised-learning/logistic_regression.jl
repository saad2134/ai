using GLM, DataFrames, Random, Statistics

function logistic_regression_student()
    Random.seed!(42)
    
    # Create student performance dataset
    n_students = 100
    data = DataFrame(
        study_hours = rand(1:20, n_students),
        attendance = rand(60:100, n_students),
        previous_gpa = rand(2.0:4.0, n_students),
        passed = zeros(Bool, n_students)  # Target: passed final exam
    )
    
    # Create realistic pass/fail pattern
    for i in 1:n_students
        pass_prob = logistic(0.1 * data.study_hours[i] + 
                            0.05 * data.attendance[i] + 
                            0.8 * data.previous_gpa[i] - 3.5)
        data.passed[i] = rand() < pass_prob
    end
    
    println("Student Performance Dataset:")
    println("Pass rate: $(mean(data.passed))")
    println("Average study hours: $(mean(data.study_hours))")
    println("Average attendance: $(mean(data.attendance))%")
    
    # Fit logistic regression model
    model = glm(@formula(passed ~ study_hours + attendance + previous_gpa), 
                data, Binomial(), LogitLink())
    
    println("\n=== Logistic Regression Results ===")
    println(model)
    
    # Predict probabilities for new students
    new_students = DataFrame(
        study_hours = [5, 15, 25],
        attendance = [70, 90, 95],
        previous_gpa = [2.5, 3.5, 4.0]
    )
    
    probabilities = predict(model, new_students)
    predictions = probabilities .> 0.5
    
    println("\nPredictions for new students:")
    for (i, (hours, att, gpa)) in enumerate(eachrow(new_students))
        prob = probabilities[i]
        pred = predictions[i]
        @printf "Student %d: %d hours, %d%%, GPA %.1f -> " i hours att gpa
        @printf "Pass probability: %.3f -> %s\n" prob (pred ? "PASS" : "FAIL")
    end
    
    return model, data
end

function logistic(x)
    return 1 / (1 + exp(-x))
end

function binary_classification_metrics(y_true, y_pred, probabilities)
    # Calculate classification metrics
    accuracy = mean(y_pred .== y_true)
    
    # Precision, Recall, F1
    tp = sum((y_pred .== true) .& (y_true .== true))
    fp = sum((y_pred .== true) .& (y_true .== false))
    fn = sum((y_pred .== false) .& (y_true .== true))
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    println("\n=== Classification Metrics ===")
    println("Accuracy:  $(round(accuracy, digits=3))")
    println("Precision: $(round(precision, digits=3))")
    println("Recall:    $(round(recall, digits=3))")
    println("F1-Score:  $(round(f1, digits=3))")
    
    return (accuracy=accuracy, precision=precision, recall=recall, f1=f1)
end

# Run logistic regression demo
logistic_model, student_data = logistic_regression_student()

# Calculate metrics on training data
probs = predict(logistic_model, student_data)
preds = probs .> 0.5
metrics = binary_classification_metrics(student_data.passed, preds, probs)