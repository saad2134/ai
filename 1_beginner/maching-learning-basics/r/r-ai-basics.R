# Basic AI and Machine Learning in R

# Load required libraries
library(caret)      # Classification and Regression Training
library(e1071)      # Naive Bayes, SVM
library(randomForest) # Random Forest
library(cluster)    # Clustering algorithms

# Set seed for reproducibility
set.seed(42)

# 1. LINEAR REGRESSION
cat("=== 1. LINEAR REGRESSION ===\n")
# Create sample data
house_data <- data.frame(
  size = c(1000, 1500, 2000, 2500, 3000, 3500),
  price = c(300000, 400000, 500000, 550000, 600000, 650000)
)

# Train linear regression model
lm_model <- lm(price ~ size, data = house_data)
print(summary(lm_model))

# Make prediction
new_house <- data.frame(size = 2800)
predicted_price <- predict(lm_model, new_house)
cat(sprintf("Predicted price for 2800 sqft: $%.2f\n", predicted_price))

# 2. K-MEANS CLUSTERING
cat("\n=== 2. K-MEANS CLUSTERING ===\n")
# Create sample customer data
customer_data <- data.frame(
  age = c(25, 35, 45, 20, 30, 40, 50, 55, 60, 32),
  income = c(30000, 45000, 60000, 25000, 35000, 70000, 80000, 90000, 95000, 48000),
  spending_score = c(70, 40, 30, 80, 60, 20, 10, 5, 2, 55)
)

# Perform K-means clustering
kmeans_result <- kmeans(customer_data, centers = 3)
customer_data$cluster <- kmeans_result$cluster

print("Cluster centers:")
print(kmeans_result$centers)
cat("Cluster sizes:", table(kmeans_result$cluster), "\n")

# 3. NAIVE BAYES CLASSIFIER
cat("\n=== 3. NAIVE BAYES CLASSIFIER ===\n")
# Create sample spam/not-spam data
email_data <- data.frame(
  word_count = c(50, 200, 30, 500, 80, 350, 45, 600),
  has_links = c(1, 1, 0, 1, 0, 1, 0, 1),
  has_urgent = c(0, 1, 0, 1, 0, 1, 0, 1),
  is_spam = c(0, 1, 0, 1, 0, 1, 0, 1)  # 0 = not spam, 1 = spam
)

# Train Naive Bayes model
nb_model <- naiveBayes(as.factor(is_spam) ~ ., data = email_data)
print(nb_model)

# Make prediction
new_email <- data.frame(word_count = 400, has_links = 1, has_urgent = 1)
prediction <- predict(nb_model, new_email)
cat("Prediction for new email:", as.character(prediction), "\n")

# 4. DECISION TREE
cat("\n=== 4. DECISION TREE ===\n")
# Create sample loan approval data
loan_data <- data.frame(
  age = c(25, 35, 45, 55, 28, 32, 40, 60),
  income = c(30000, 50000, 80000, 60000, 35000, 45000, 70000, 90000),
  credit_score = c(650, 700, 750, 680, 720, 690, 780, 710),
  approved = c(0, 1, 1, 1, 0, 1, 1, 1)  # 0 = denied, 1 = approved
)

# Train decision tree
library(rpart)
tree_model <- rpart(as.factor(approved) ~ ., data = loan_data, method = "class")
print(tree_model)

# Visualize tree (optional - uncomment to see plot)
# library(rpart.plot)
# rpart.plot(tree_model)

# 5. RANDOM FOREST
cat("\n=== 5. RANDOM FOREST ===\n")
# Use iris dataset for classification
data(iris)

# Split data
train_index <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
train_data <- iris[train_index, ]
test_data <- iris[-train_index, ]

# Train Random Forest
rf_model <- randomForest(Species ~ ., data = train_data, ntree = 100)
print(rf_model)

# Make predictions
predictions <- predict(rf_model, test_data)
confusion_matrix <- table(Predicted = predictions, Actual = test_data$Species)
print("Confusion Matrix:")
print(confusion_matrix)

accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat(sprintf("Accuracy: %.2f%%\n", accuracy * 100))

# 6. PRINCIPAL COMPONENT ANALYSIS (PCA)
cat("\n=== 6. PRINCIPAL COMPONENT ANALYSIS ===\n")
# Perform PCA on iris data
pca_result <- prcomp(iris[, 1:4], scale = TRUE)
print(summary(pca_result))

# Plot PCA results (first two components)
plot(pca_result$x[, 1:2], col = as.numeric(iris$Species), 
     pch = 19, main = "PCA of Iris Dataset")
legend("topright", legend = levels(iris$Species), 
       col = 1:3, pch = 19)

cat("\n=== R AI DEMO COMPLETE ===\n")