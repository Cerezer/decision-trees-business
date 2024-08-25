# Install and load necessary packages
#install.packages(c("rpart", "caret", "ggplot2"))
library(rpart)
library(caret)
library(ggplot2)

# Set seed for reproducibility
set.seed(42)

# Step 1: Generate Fake Data
n <- 200  # Number of observations
data <- data.frame(
  Age = sample(18:70, n, replace = TRUE),
  Income = sample(20000:120000, n, replace = TRUE),
  SpendingScore = sample(1:100, n, replace = TRUE),
  Segment = sample(c("Low", "Medium", "High"), n, replace = TRUE)
)

# View the first few rows of the data
head(data)

# Step 2: Split the Data into Training and Testing Sets
set.seed(42)
trainIndex <- createDataPartition(data$Segment, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Step 3: Build the Decision Tree Model
treeModel <- rpart(Segment ~ Age + Income + SpendingScore, 
                   data = trainData, 
                   method = "class")

# Print the model summary
summary(treeModel)

# Step 4: Visualize the Decision Tree
plot(treeModel)
text(treeModel, use.n = TRUE, all = TRUE, cex = 0.5)

# Step 5: Predict and Evaluate the Model

# Predict on the test data
predictions <- predict(treeModel, newdata = testData, type = "class")

testData$Segment <- factor(testData$Segment) # Ensure the reference variable is a factor

predictions <- factor(predictions, levels = levels(testData$Segment)) # Ensure predictions are a factor with the same levels as the reference

confusionMatrix(predictions, testData$Segment) # Generate the confusion matrix


# Step 6: Fine-Tune the Model with caret
# Define the training control
train_control <- trainControl(method = "cv", number = 10)

# Train the model with caret
tunedModel <- train(Segment ~ Age + Income + SpendingScore, 
                    data = trainData, 
                    method = "rpart", 
                    trControl = train_control)

# Print the best model
print(tunedModel$finalModel)

# Step 7: Visualize Model Performance with ggplot2
# Plot the complexity parameter (cp) vs accuracy
ggplot(tunedModel) +
  geom_line(aes(x = tunedModel$results$cp, y = tunedModel$results$Accuracy)) +
  labs(x = "Complexity Parameter (cp)", y = "Accuracy") +
  ggtitle("Model Accuracy vs Complexity Parameter")

