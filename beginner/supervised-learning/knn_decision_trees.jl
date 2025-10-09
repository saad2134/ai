using MLJ, DataFrames, Random, Statistics

function knn_iris_demo()
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.features, iris.targets
    
    println("Iris Dataset:")
    println("Features: $(names(X))")
    println("Target classes: $(unique(y))")
    println("Dataset size: $(size(X, 1)) samples")
    
    # Convert to DataFrame for easier handling
    data = DataFrame(X)
    data.species = y
    
    # Split data
    train_idx, test_idx = partition(eachindex(y), 0.7, shuffle=true, rng=42)
    
    X_train = X[train_idx, :]
    X_test = X[test_idx, :]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    # K-Nearest Neighbors classifier
    KNN = @load KNNClassifier pkg=NearestNeighborModels
    knn_model = KNN()
    
    # Fit and predict
    mach = machine(knn_model, X_train, y_train)
    fit!(mach)
    y_pred = predict(mach, X_test)
    
    # Evaluate
    accuracy = mean(y_pred .== y_test)
    println("\n=== K-Nearest Neighbors Results ===")
    println("Accuracy: $(round(accuracy, digits=3))")
    
    # Confusion matrix
    cm = confusion_matrix(y_pred, y_test)
    println("Confusion Matrix:")
    println(cm)
    
    return knn_model, X, y
end

function decision_tree_titanic()
    # Create Titanic-like dataset
    Random.seed!(42)
    n_passengers = 200
    
    data = DataFrame(
        pclass = rand([1, 2, 3], n_passengers),      # Passenger class
        sex = rand(["male", "female"], n_passengers), # Gender
        age = rand(1:80, n_passengers),              # Age
        sibsp = rand(0:3, n_passengers),             # Siblings/Spouse
        fare = rand(10:200, n_passengers),           # Fare
        survived = zeros(Bool, n_passengers)         # Target
    )
    
    # Create survival patterns (simplified)
    for i in 1:n_passengers
        survival_prob = 0.0
        if data.sex[i] == "female"
            survival_prob += 0.6
        end
        if data.pclass[i] == 1
            survival_prob += 0.3
        elseif data.pclass[i] == 3
            survival_prob -= 0.2
        end
        if data.age[i] < 18
            survival_prob += 0.2
        end
        
        data.survived[i] = rand() < survival_prob
    end
    
    # Convert categorical variables
    data.sex = map(x -> x == "female" ? 1 : 0, data.sex)
    
    println("Titanic-like Dataset:")
    println("Survival rate: $(mean(data.survived))")
    println("Survival by class:")
    println(combine(groupby(data, :pclass), :survived => mean))
    
    # Prepare features and target
    features = select(data, [:pclass, :sex, :age, :sibsp, :fare])
    target = data.survived
    
    # Split data
    train_idx, test_idx = partition(eachindex(target), 0.7, shuffle=true, rng=42)
    
    X_train = features[train_idx, :]
    X_test = features[test_idx, :]
    y_train = target[train_idx]
    y_test = target[test_idx]
    
    # Decision Tree classifier
    Tree = @load DecisionTreeClassifier pkg=DecisionTree
    tree_model = Tree(max_depth=4)
    
    mach = machine(tree_model, X_train, y_train)
    fit!(mach)
    y_pred = predict(mach, X_test)
    
    # Evaluate
    accuracy = mean(y_pred .== y_test)
    println("\n=== Decision Tree Results ===")
    println("Accuracy: $(round(accuracy, digits=3))")
    
    # Feature importance (simplified)
    println("\nFeature Importance (based on tree structure):")
    feature_names = names(features)
    println("The tree uses features in this order of importance based on splits")
    
    return tree_model, data
end

# Run the functions
knn_model, iris_X, iris_y = knn_iris_demo()
tree_model, titanic_data = decision_tree_titanic()