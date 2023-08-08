#install and load packages using install.packages & library fx
packagesNeeded = c("caret", "e1071", "mlbench", "party", "ipred",
                   "randomForest", "gbm", "Cubist")
install.packages(packagesNeeded)

library(caret)
library(e1071)
library(mlbench)
library(party)
library(ipred)
library(randomForest)
library(gbm)
library(Cubist)

# ---------------------------------------------------------------------------
#Prepare training and test datasets
#Using the BostonHousing data, part of mlbench library
data(BostonHousing) #load the dataset
BH = BostonHousing #assign to new object for manipulation

#Now using the createDataPartition fx to split into train and test sets
index = createDataPartition(BH$medv, p = 0.80, list = FALSE)
#Assign new index object to split BH into train & test
BHtrain = BH[index,]
BHtest = BH[-index,]
#Data has been split and is now prepared for modeling


# ---------------------------------------------------------------------------
#Conditional Inference Tree
# ---------------------------------------------------------------------------

#Use ctree() from the party library to create the conditional inference tree
cit_model = ctree(medv ~ ., data = BHtrain)
cit_model #observe the model that was created

#Use the predict() function to run our model our on test dataset BHtest
cit_pred = predict(cit_model, BHtest)

#Create a data frame of our observed and predicted medv from the model
cit_df = data.frame(obs = BHtest$medv, pred = cit_pred[,])

#lastly use defaultSummary() on the combined dataframe to evaluate the 
#model's performance on the test dataset
cit_perf = defaultSummary(cit_df)
cit_perf


# ---------------------------------------------------------------------------
#Bagged Tree
# ---------------------------------------------------------------------------

#Using bagging() from ipred library to create bootstrap aggregated (bagged) 
#decision trees on the BostonHousing data
bag_model = bagging(medv ~ ., data = BHtrain, nbagg = 100, coob = TRUE)
#^^setting nbagg to generate 100 trees and compute the Out of Bag error rate
bag_model #observe the model, OOB error = 4.1392

#Use the predict() function to run our model our on test dataset BHtest
bag_pred = predict(bag_model, BHtest)

#Create a data frame of our observed and predicted medv from the model
bag_df = data.frame(obs = BHtest$medv, pred = bag_pred)

#lastly use defaultSummary() on the combined dataframe to evaluate the 
#model's performance on the test dataset
bag_perf = defaultSummary(bag_df)
bag_perf


# ---------------------------------------------------------------------------
#Random Forest
# ---------------------------------------------------------------------------

#Using randomForest() from the randomForest library to create our model
rf_model = randomForest(medv ~ ., data = BHtrain, mtry = 4) 
#^^^ setting mtry = 4 because this is a regression with 13 predictors, using
# m = p/3 for regression gives us 13/3 = 4.133 = m
rf_model #exmaine the model details
sqrt(mean(rf_model$mse)) #RMSE = 3.363329
mean(rf_model$rsq) #R-squared = 0.8707791

#Now we can use train() to tune and try to optimize mtry for our randomForest
rf_train = train(medv ~ ., data = BHtrain, method = "rf", 
                 trControl = trainControl(method = "cv", number = 10))
rf_train$bestTune #see optimal value for mtry
#Best mtry value is 7
rf_train #examine model details
#RMSE = 3.146808, Rsquared = 0.8913780

#Non-tuned RF model performed well but the tuned rf_train model was better,
#we will use the tuned model to make our predictions on the test data

#Again using predict() to test our model on the BHtest dataset
rf_pred = predict(rf_train, BHtest)

#Create a data frame comparing our observed and predicted medv values
rf_df = data.frame(obs = BHtest$medv, pred = rf_pred)

#Use defaultSummary() to measure the model's performance on the test data
rf_perf = defaultSummary(rf_df)
rf_perf


# ---------------------------------------------------------------------------
#Conditional Inference Forest
# ---------------------------------------------------------------------------

#Use the cforest() function from the party library to build conditional 
#inference forests
cif_model = cforest(medv ~ ., data = BHtrain)
cif_model 

#Now we will make predictions on the test data using our model and predict()
cif_pred = predict(cif_model, newdata = BHtest, OOB = TRUE) 
#predict.randomForest requires you specify the newdata being used,
#otherwise it will not use the test data to make predictions
#ALSO this function will not run without OOB = TRUE added to predict function, 
#seems to be a bug with this particular model and predict(). Kept returning
#this error message: Error in OOB && is.null(newdata) : invalid 'x' type in 'x && y'
#Solution found @ https://groups.google.com/g/rattle-users/c/33AHXWrP5Vc?pli=1

#Now our predict function works, we can compare our predicted values
#Using data.frame() to create a dataframe of the observed and predicted medv
cif_df = data.frame(obs = BHtest$medv, pred = cif_pred[,]) #remove header

#Now use defaultSummary() to evaluate the model's performance on BHtest
cif_perf = defaultSummary(cif_df)
cif_perf


# ---------------------------------------------------------------------------
#Gradient Boosted Tree
# ---------------------------------------------------------------------------

#Use the gbm() function from the gmb library to create our boosted trees
gbm_model = gbm(medv ~ ., data = BHtrain, distribution = "gaussian")
#specifying Gaussian because this is a regression problem, note from ?gbm
#says model will assume which to use based on data but since we know the data
#is continuous we can specify which method to use
gbm_model #examine our model's results 
#100 trees were computed, and 11 of 13 predictors had a non-zero influence

#Now we have our model, we can make predictions on the test data
gbm_pred = predict(gbm_model, BHtest)

#Now we will create a data frame to house the observed and predicted values
gbm_df = data.frame(obs = BHtest$medv, pred = gbm_pred)

#Lastly, use defaultSummary() to obtain the model's performance on BHtest
gbm_perf = defaultSummary(gbm_df)
gbm_perf


# ---------------------------------------------------------------------------
#Cubist
# ---------------------------------------------------------------------------

#For our last model, we will use the cubist() function from the cubist library
#Cubist is a bit different from the other regression trees we've developed, it is
#a rule based model, and for its arguments we have to specify the x and y values
#to use to create it
x = BHtrain[,1:13] #our 13 predictors
y = BHtrain[,14] #medv the outcome variable
cube_model = cubist(x,y) #defaulting to 1 committee in our model (no boosting)
cube_model #with our 1 committee the model found 7 rules
summary(cube_model)
#Average Absolute Error: 2.09, Rules = 7


#As noted in the help file ?cubist, they recommend using train() to tune the
#model's number of committees and neighbors. So we will do that see how much it
#improves the cubist model
cube_train = train(medv ~ ., data = BHtrain, method = "cubist", 
                   trControl = trainControl(method = "cv", number = 10))
cube_train$bestTune #see what our optimized committee and neighbors are
#found that committees = 20, and neighbors = 5
cube_train
#using summary() we see that cube_train found 20 rules compared to cube_model
summary(cube_train)
#Average Absolute Error: 1.83, Rules = 20

#Our cube_train model performed significantly better on the training data, we
#will use to make our predictions on the test data
cube_pred = predict(cube_train, BHtest)

#Now create a data frame of our observed and predicted medv values
cube_df = data.frame(obs = BHtest$medv, pred = cube_pred)

#lastly use defaultSummary() to evaluate its performance on the test data
cube_perf = defaultSummary(cube_df)
cube_perf

# ---------------------------------------------------------------------------
#Using rbind() to combine and compare all of the model's performance on the
#test data
rbind("Conditional Inference Tree" = cit_perf,
      "Bagged Tree" = bag_perf,
      "Random Forest" = rf_perf,
      "Conditional Inference Forest" = cif_perf,
      "Gradient Boosted Tree" = gbm_perf,
      "Cubist" = cube_perf)
#Cubist is our best model to use for the BostonHousing dataset
