# ----------------------
# QUMT 6350 Homework 4
# Thomas Crawford

#install and load packages using install.packages & library fx
packagesNeeded = c("caret", "e1071", "pROC", "MASS", "glmnet",
                   "modeldata")
install.packages(packagesNeeded)

library(caret)
library(e1071)
library(pROC)
library(MASS)
library(glmnet)
library(modeldata)

# ---------------------------------------------------------------------------
#Prepare training and test datasets
#Using the mlc_churn data, part of the modeldata library
data("mlc_churn") #load the dataset
mlc = mlc_churn #assign to new object for manipulation
View(mlc) #examine the dataset
str(mlc) #looking at the structure
sum(is.na(mlc)) #check for any null values

#Using set.seed() so we can replicate our results
set.seed(111)

#Now using the createDataPartition fx to split into train and test sets
index = createDataPartition(mlc$churn, p = 0.80, list = FALSE)
#Assign new index object to split mlc into train & test
mlctrain = mlc[index,]
mlctest = mlc[-index,]
#Data has been split and is now prepared for modeling


# ---------------------------------------------------------------------------
#Logistic Regression
# ---------------------------------------------------------------------------

#Using the glm() function to create our logistic regression model
glm_model = glm(churn ~ ., family = binomial, data = mlctrain)
summary(glm_model) #examine the results of the model fitting
varImp(glm_model) #examine importance of certain variables in the model
#Our most important variables are "International PlanYes", "Number Customer 
#Service Calls" and "Voice Mail Plan Yes"

#Now we've created and examine the model, we can run predictions on the testset
glm_pred = predict(glm_model, mlctest, type = "response")

#We've now created our predicted probablities of each class, now need to convert
#into factors so we can create our confusionMatrix
#saving as a new object to not interfere with original stored predictions for roc()
glm_predf = ifelse(glm_pred<0.5, "yes", "no") #turning probabilities into labels
glm_pred_factor = factor(glm_predf, levels = c("yes", "no")) #converting labels into factors

#Now we can create a confusion matrix to evaluate the model's performance on the
#test dataset
glm_cm = confusionMatrix(glm_pred_factor, mlctest$churn)
glm_cm #view our confusionMatrix

#now we will create our ROC curve using roc()
#Need to use levels() and rev() to switch order of labels, as roc will assume
#that "yes" is the class of interest when we are interested "no"
glm_roc = roc(response = mlctest$churn, predictor = glm_pred, 
              levels = rev(levels(mlctest$churn)))
#Now we can plot the ROC 
plot(glm_roc, main = "ROC Curve for Logistic Regression")


# ---------------------------------------------------------------------------
#Linear Discriminant Analysis
# ---------------------------------------------------------------------------

#Using the lda() function to create our linear discriminant model
lda_model = lda(churn ~ ., data = mlctrain)
lda_model
summary(lda_model) #view our model's results
#~14% of records had churn and 85% did not

#now we can make predictions using the model on the test data
lda_pred = predict(lda_model, mlctest, type = "response")

#Now we have our predictions, we can create a confusionmatrix to measure the 
#performance of the model compared to the test set
lda_cm = confusionMatrix(lda_pred$class, mlctest$churn)
lda_cm #run to view the table and it's statistics

#Now we will create the Receiver Operator Characteristics (ROC) plot
lda_roc = roc(response = mlctest$churn, predictor = lda_pred$posterior[,1])
#Now we can plot our ROC curve
plot(lda_roc, main = "ROC Curve for LDA")


# ---------------------------------------------------------------------------
#Penalized Model - Using Lasso Logistic Regression
# ---------------------------------------------------------------------------

#Using the glmnet() function to create our penalized lasso regression model
#glmnet() requires the x and y variables be specified
x = as.matrix(mlctrain) #glmnet requires x argument be a matrix 
y = mlctrain$churn #specifying our outcome variable

#Using glmnet to create our lasso model, tuning the model by specifying a  
#value for lambda, which are going to set at 0.001
lasso_model = glmnet(x, y, alpha = 1, family = "binomial",
                     lambda = 0.001)
summary(lasso_model) #view the results of our model
varImp(lasso_model, lambda = 0.001) #view the variable importance of predictors

#Now we have created the model, we can run our predictions on the test data
#have to set our test dataset as a matrix because the training data was converted
#to a matrix when creating the model
lasso_pred = predict(lasso_model, data.matrix(mlctest), type = "response")

#Lasso is also a logistic regression, and like above, we need to convert the
#model predictions/probabilities into factors for our confusionmatrix to use
#saving as a new object to not interfere with original stored predictions for roc()
lasso_predf = ifelse(lasso_pred<0.5, "yes", "no") #turning probabilities into labels
lasso_pred_factor = factor(lasso_predf, levels = c("yes", "no")) #converting labels into factors

#Now we can create the confusionMatrix to evaluate our model's performance
lasso_cm = confusionMatrix(data = lasso_pred_factor, reference = mlctest$churn)
lasso_cm

#Now we will create the ROC curve from the model predctions
#Again since lasso is also a logistic regression, we will need to reverse
#the class labels so "no" is our class of interest
lasso_roc = roc(response = mlctest$churn, predictor = lasso_pred[,1],
                levels = rev(levels(mlctest$churn)))
#now we can plot the ROC curve
plot(lasso_roc, main = "ROC Curve for Lasso Regression")


# ---------------------------------------------------------------------------
#K-Nearest Neighbors (Classification)
# ---------------------------------------------------------------------------

#Using the train() function to create our KNN model
#using trainControl() to specify using 10-fold cross validation for tuning
knn_model = train(churn ~ ., data = as.data.frame(mlctrain), method = "knn",
                  trControl = trainControl(method = "cv", number = 10, classProbs = TRUE))
knn_model
summary(knn_model) #exmaine the results of our model
varImp(knn_model) #see what are our important variables in the model

#Using the predict() with our model and the test dataset
#using "type =" to generate probabilities
knn_pred = predict(knn_model, mlctest, type = "prob")

#Need to convert the predicted probabilities into factors for the confusionMatrix
#saving as a new object to not interfere with original stored predictions for roc()
knn_predf = ifelse(knn_pred[,2]<0.5, "yes", "no")
knn_pred_factor = factor(knn_predf, levels = c("yes", "no"))

#Now we create a confusionMatrix to evaluate the model's predictive performance
knn_cm = confusionMatrix(knn_pred_factor, mlctest$churn)
knn_cm #view our confusionMatrix and its stats

#Now we will create a ROC curve 
#Again need to tell roc which class is the outcome of interest ("no"), we
#tell roc() this by specifying the levels and reversing the levels from the test
knn_roc = roc(response = mlctest$churn, predictor = knn_pred[,2],
              levels = rev(levels(mlctest$churn)))
#Now we can plot our ROC curve
plot(knn_roc, main = "ROC Curve for K-Nearest Neighbor")
