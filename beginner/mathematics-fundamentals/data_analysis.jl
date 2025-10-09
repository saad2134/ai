using DataFrames, CSV, Statistics

function analyze_titanic()
    # Create sample Titanic-like dataset
    data = DataFrame(
        PassengerId = 1:10,
        Survived = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
        Pclass = [3, 1, 2, 3, 3, 1, 2, 2, 1, 3],
        Name = ["Braund, Mr. Owen", "Cumings, Mrs. John", "Heikkinen, Miss. Laina", 
                "Futrelle, Mrs. Jacques", "Allen, Mr. William", "Moran, Mr. James",
                "McCarthy, Mr. Timothy", "Palsson, Master. Gosta", "Johnson, Mrs. Oscar", 
                "Nasser, Mrs. Nicholas"],
        Sex = ["male", "female", "female", "female", "male", "male", "male", "male", "female", "female"],
        Age = [22, 38, 26, 35, 35, missing, 54, 2, 27, 14],
        Fare = [7.25, 71.28, 7.92, 53.10, 8.05, 15.50, 51.86, 21.08, 41.58, 30.07]
    )
    
    println("Dataset Info:")
    println("Shape: $(size(data))")
    println("\nFirst 5 rows:")
    show(first(data, 5), allcols=true)
    println("\n")
    
    # Data cleaning
    println("Data Cleaning:")
    println("Missing values in Age: $(sum(ismissing.(data.Age)))")
    
    # Fill missing ages with mean
    age_mean = mean(skipmissing(data.Age))
    data.Age = coalesce.(data.Age, age_mean)
    println("After filling missing values:")
    
    # Basic statistics
    println("\nBasic Statistics:")
    println("Survival rate: $(mean(data.Survived))")
    println("Average age: $(mean(data.Age))")
    println("Average fare: $(mean(data.Fare))")
    
    # Group by class
    println("\nSurvival by Class:")
    class_survival = combine(groupby(data, :Pclass), :Survived => mean)
    println(class_survival)
    
    return data
end

analyze_titanic()