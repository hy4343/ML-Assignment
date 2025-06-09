# Below is the R code which could be implemented in R studio directly

# Part 1: Build and train the machine learning model

# Step 1: Install and Load related packages in the environment

install.packages("caret")
install.packages("randomForest")
library(caret)
library(randomForest)

# Step 2: Load the dataset (Here, I have downloaded it on my desktop)

training <- read.csv("~/Desktop/training.csv", na.strings = c("NA", ""))

# Step 3: Clean the dataset

nzv <- nearZeroVar(training)
training <- training[, -nzv] # Remove near-zero variance columns, meaningless for training the model

training <- training[, colSums(is.na(training)) == 0] # Remove columns without values

training <- training[, -c(1:5)] # Remove irrelevant columns (username.etc)

# Step 4: Split into training and testing datasets

set.seed(125)
inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE) # While preserving class distrivution, create a 70% training / 30% testing dataset
trainSet <- training[inTrain, ]
testSet  <- training[-inTrain, ]

# Step 5: Train the random forest model
# Here, I use the 5-fold cross-validation since it is both faster and relatively reliable. As my Mac trains the model not so fast,I choose the 5-fold instead of the 10-fold.

control <- trainControl(method = "cv", number = 5)

modelFit <- train(
  classe ~ ., 
  data = trainSet, 
  method = "rf", 
  trControl = control,
  importance = TRUE,
  ntree = 100   # Train only 100 trees instead of 500 (default) to improve the training speed
)

# Step 6: Make predictions and evaluate the model accuracy

predictions <- predict(modelFit, testSet)
confusionMatrix(predictions, testSet$classe)

# The final result implicates that the model correctly classifies the exercise type (classe) in 99.61% of cross-validated training samples, which suggests reliable.


# Part 2: Applying the machine learning algorithm to test cases

# Load the dataset
testData <- read.csv("~/Desktop/test.csv", na.strings = c("NA", ""))

# Clean the dataset to follow the format in the model
testData <- testData[, -nzv]
testData <- testData[, colSums(is.na(testData)) == 0]
testData <- testData[, -c(1:5)]

# Predict and see the result level 
predictedClasses <- predict(modelFit, testData)
print(predictedClasses)
