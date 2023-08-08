#install and load packages using install.packages & library fx
packagesNeeded = c("caret", "e1071", "pROC", "rpart", "rpart.plot",
                   "modeldata", "ipred", "randomforest", "C50")
install.packages(packagesNeeded)

library(caret)
library(e1071)
library(pROC)
library(rpart)
library(rpart.plot)
library(modeldata)
library(ipred)
library(randomForest)
library(C50)


# ---------------------------------------------------------------------------
#Prepare training and test datasets
#Using the mlc_churn data, part of the modeldata library
data("mlc_churn") #load the dataset
mlc = mlc_churn #assign to new object for manipulation
str(mlc) #looking at the structure
sum(is.na(mlc)) #check for any null values

#removing state, area_code, account_length, total_night_calls, and total_eve_calls
#from our mlc dataset as they are not needed for our analysis
mlc_new = mlc[,-c(1,2,3,11,14)]
str(mlc_new) #verify columns have been removed

#Using set.seed() so we can replicate our results
set.seed(111)

#Now using the createDataPartition fx to split into train and test sets
index = createDataPartition(mlc_new$churn, p = 0.80, list = FALSE)
#Assign new index object to split mlc into train & test
mlctrain = mlc_new[index,]
mlctest = mlc_new[-index,]
#Data has been split and is now prepared for modeling


# ---------------------------------------------------------------------------
#Classification and Regression Tree
# ---------------------------------------------------------------------------

#Using the rpart() function to create our CART model on the training set
#Tuning the model by specifying a complexity parameter in the rpart.control()
cart_model = rpart(churn ~ ., data = mlctrain, method = "class",
                   control = rpart.control(cp = 0.01))
cart_model$variable.importance #observe model's important variables
#Our most influential variables are "total_day_minutes" and "total_day_charge"

#Now we can make predictions for our confusion matrix
cart_pred1 = predict(cart_model, newdata = mlctest, type = "class")
#Now we construct our confusion matrix
cart_cm = confusionMatrix(cart_pred1, mlctest$churn)
cart_cm

#Now we will make predictions for our ROC curve calculating probabilities
cart_pred2 = predict(cart_model, newdata = mlctest, type = "prob")
#Now we combine our class predictions with our probability predictions using cbind
cart_combine = cbind(as.data.frame(cart_pred1), cart_pred2)
#Now we can create the ROC curve and plot it
cart_roc = roc(response = mlctest$churn, predictor = cart_combine$yes)
plot(cart_roc, main = "ROC Curve for CART Model")

#Gather model metrics for comparison at the end:
cart_metrics = data.frame(Accuracy = cart_cm$overall["Accuracy"],
                         Sensitivity = cart_cm$byClass["Sensitivity"],
                         specificity = cart_cm$byClass["Specificity"],
                         ROC_AUC = auc(cart_roc),
                         row.names = "CART")
cart_metrics
# ---------------------------------------------------------------------------
#Bagged Trees
# ---------------------------------------------------------------------------

#Using bagging() function to create our bagged tree model
#also using rpart.control() to use 10-fold cross-validation
bag_model = bagging(churn ~ ., data = mlctrain, coob = TRUE,
                    control = rpart.control(xval = 10))
varImp(bag_model) #check what variables are most important
#Just like our CART model "total_day_charge" and "total_day_minutes" are
#the most significant variables for our model

#Now we will create predictions so we can create our confusion matrix
bag_pred1 = predict(bag_model, newdata = mlctest, type = "class")
#Constructing the confusion matrix
bag_cm = confusionMatrix(bag_pred1, mlctest$churn)
bag_cm

#Next we will create probability predictions with our model
bag_pred2 = predict(bag_model, newdata = mlctest, type = "prob")
#Before creating our ROC curve, we need to combine our class and prob 
#predictions into a single dataframe - to do this we will use cbind()
bag_combine = cbind(as.data.frame(bag_pred1), bag_pred2)
#Now we can create our ROC and plot it
bag_roc = roc(response = mlctest$churn, predictor = bag_combine$yes)
plot(bag_roc, main = "ROC Curve for Bagged Trees")

#Gather model metrics for comparison at the end:
bag_metrics = data.frame(Accuracy = bag_cm$overall["Accuracy"],
                          Sensitivity = bag_cm$byClass["Sensitivity"],
                          specificity = bag_cm$byClass["Specificity"],
                          ROC_AUC = auc(bag_roc),
                          row.names = "Bagged Trees")
bag_metrics

# ---------------------------------------------------------------------------
#Random Forest
# ---------------------------------------------------------------------------

#To create our randomForest, we will use the wonderful train() function from
#the caret package. We will also use the trainControl() function to control our
#model turning, specifying to use 10-fold cross validation & random search
rf_tune = trainControl(method = "cv", number = 10, search = "random")
rf_model = train(churn ~ ., data = mlctrain, method = "rf",
                 trControl = rf_tune)
varImp(rf_model) #check what variables are most important to our model
#Same our previous models: "total_day_minutes" & "total_day_charge"

#Now we've created the model, we will create class predictions for our
#confusion matrix
rf_pred1 = predict(rf_model, newdata = mlctest, type = "raw")
#Next create our confusion matrix
rf_cm = confusionMatrix(rf_pred1, mlctest$churn)
rf_cm

#Next we'll create probability predictions from our rf_model
rf_pred2 = predict(rf_model, newdata = mlctest, type = "prob")
#Next we will combine our class and prob predictions using cbind so we can
#use the predicted resutls to create our ROC curve
rf_combine = cbind(as.data.frame(rf_pred1), rf_pred2)
#Now we can calculate our ROC curve and plot it
rf_roc = roc(response = mlctest$churn, predictor = rf_combine$yes)
plot(rf_roc, main = "ROC Curve for Random Forest")

#Gather model metrics for comparison at the end:
rf_metrics = data.frame(Accuracy = rf_cm$overall["Accuracy"],
                          Sensitivity = rf_cm$byClass["Sensitivity"],
                          specificity = rf_cm$byClass["Specificity"],
                          ROC_AUC = auc(rf_roc),
                          row.names = "Random Forest")
rf_metrics

# ---------------------------------------------------------------------------
#C5.0
# ---------------------------------------------------------------------------

#For our last model we will use the C5.0() function to create our C5.0 model
#we will tune the model using C5.0Control() by setting winnow = TRUE so the model
#uses feature selection on our predictors
c50_model = C5.0(churn ~ ., data = mlctrain, 
                 control = C5.0Control(winnow = TRUE))
C5imp(c50_model) #observe our most important variables
#Different influential variables compared to our previous model, as "total_day_charge"
#is not an important variable in this model. While "total_day_minutes" still is

#Next we create class predictions and then evaluate the model using a confusion matrix
c50_pred1 = predict(c50_model, newdata = mlctest, type = "class")
c50_cm = confusionMatrix(c50_pred1, mlctest$churn)
c50_cm

#Now we'll create probability predictions, and then combine that our class 
#predictions into a single data frame
c50_pred2 = predict(c50_model, newdata = mlctest, type = "prob")
c50_combine = cbind(as.data.frame(c50_pred1), c50_pred2)
#We've created the combined data frame of our predictions, now we calculate
#our ROC curve and plot it
c50_roc = roc(response = mlctest$churn, predictor = c50_combine$yes)
plot(c50_roc, main = "ROC Curve for C5.0 Model")

#Gather model metrics for comparison at the end:
c50_metrics = data.frame(Accuracy = c50_cm$overall["Accuracy"],
                          Sensitivity = c50_cm$byClass["Sensitivity"],
                          specificity = c50_cm$byClass["Specificity"],
                          ROC_AUC = auc(c50_roc),
                          row.names = "C5.0")
c50_metrics

# ---------------------------------------------------------------------------
#Comparing model performance on the mlc_churn dataset
# ---------------------------------------------------------------------------

#Create combined dataframe of all our model's confusion matrices and ROC curves
rbind(cart_metrics, bag_metrics, rf_metrics, c50_metrics)

#Our most accurate model on the Random Forest, which also had the highest AUC
#score of the 4 models.

#Creating a plot showing the ROC curves for all 4 models
plot(cart_roc, col = "blue", lty = 1, main = "ROC Curve Comparison")
plot(bag_roc, add = TRUE, col = "red", lty = 2)
plot(rf_roc, add = TRUE, col = "green", lty = 1)
plot(c50_roc, add = TRUE, col = "orange",lty = 2)
legend(x = "bottomright",legend = c("CART","Bagged Tree","Random Forest", "C5.0"),
       lty = 1:2,col = c("blue","red","green","orange"),cex = 0.55)
#Our plotted ROC curves match the results from above, Random Forest has the 
#largest AUC score of the 4 models
