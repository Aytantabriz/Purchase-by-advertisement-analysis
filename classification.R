#================================================== CLASSIFICATION MODELS ===============================================

# **** ==================================================================================================================
setwd("C:/Users/a.huseynli/Desktop/ML/R DA/R_1_3")
# * Importing the dataset -----------------------------
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[2:5]
str(dataset)

# * Encoding features -----------------------------
dataset$Gender = ifelse(dataset$Gender=="Male",0,1)
dataset$Gender = as.factor(dataset$Gender)
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# * Splitting the dataset into the Training set and Test set -----------------------------
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# * Feature Scaling  -----------------------------
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])

# **** ==================================================================================================================
# Logistic Regression ==================================================

logistic = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set); summary(logistic)

# KNN ==================================================================
library(class)
KNN = knn(train = training_set[, -4],
             test = test_set[, -4],
             cl = training_set[, 4],
             k = 5,
             prob = TRUE)

# SVM ==================================================================
library(e1071)
SVM = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Kernel SVM ===========================================================
kernel_SVM = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Naive Bayes ==========================================================
naive_bayes = naiveBayes(x = training_set[-4],
                        y = training_set$Purchased)

# Decision Tree ========================================================
library(rpart)
decision_tree = rpart(formula = Purchased ~ .,
                   data = training_set)

# Random Forest ========================================================
library(randomForest)
set.seed(123)
random_forest = randomForest(x = training_set[-4],
                          y = training_set$Purchased,
                          ntree = 100)

# **** ==================================================================================================================
# * Predicting test set -----------------------------
library(tidyverse)

# logistics
logistic_pred = predict(logistic, type = 'response', newdata = test_set[-4])
logistic_pred = ifelse(logistic_pred > 0.5, 1, 0) %>% as.factor()

# KNN
KNN_pred <- KNN

#SVM
SVM_pred = predict(SVM, newdata = test_set[-4])

# Kernel SVM
kernel_SVM_pred = predict(kernel_SVM, newdata = test_set[-4])

# Naive Bayes
naive_bayes_pred = predict(naive_bayes, newdata = test_set[-4])

# Decision Tree
decision_tree_pred = predict(decision_tree, newdata = test_set[-4], type = 'class')

# Random Forest
random_forest_pred = predict(random_forest, newdata = test_set[-4])


# * Confusion matrix and Accuracy --------------------------------
library(caret)    

# logistics
logistics_cm <- confusionMatrix(data=logistic_pred, reference=test_set$Purchased)
logistics_accuracy <- round(logistics_cm$overall[1],2);logistics_accuracy #0.82

# KNN
KNN_cm <- confusionMatrix(data=KNN_pred, reference=test_set$Purchased)
KNN_accuracy <- round(KNN_cm$overall[1],2);KNN_accuracy #0.88

# SVM
SVM_cm <- confusionMatrix(data=SVM_pred, reference=test_set$Purchased)
SVM_accuracy <- round(SVM_cm$overall[1],2);SVM_accuracy #0.8

# Kernel SVM
kernel_SVM_cm <- confusionMatrix(data=kernel_SVM_pred, reference=test_set$Purchased)
kernel_SVM_accuracy <- round(kernel_SVM_cm$overall[1],2);kernel_SVM_accuracy #0.89

# Naive Bayes
naive_bayes_cm <- confusionMatrix(data=naive_bayes_pred, reference=test_set$Purchased)
naive_bayes_accuracy <- round(naive_bayes_cm$overall[1],2);naive_bayes_accuracy #0.87

# Decision Tree
decision_tree_cm <- confusionMatrix(data=decision_tree_pred, reference=test_set$Purchased)
decision_tree_accuracy <- round(decision_tree_cm$overall[1],2);decision_tree_accuracy # 0.83

# Random Forest
random_forest_cm <- confusionMatrix(data=random_forest_pred, reference=test_set$Purchased)
random_forest_accuracy <- round(random_forest_cm$overall[1],2);random_forest_accuracy #0.88

# * Best model selection --------------------------------

accuracy_table <- data.frame(model = c('logistic','KNN','SVM','kernel_SVM','naive_bayes','decision_tree','random_forest'),
           Accuracy = c(logistics_accuracy[[1]], KNN_accuracy[[1]], SVM_accuracy[[1]], 
                        kernel_SVM_accuracy[[1]], naive_bayes_accuracy[[1]], 
                        decision_tree_accuracy[[1]], random_forest_accuracy[[1]]));accuracy_table

accuracy_table$Accuracy

best_model <- accuracy_table[which.max(accuracy_table$Accuracy),][[1]]


