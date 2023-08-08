# ----------------------
# QUMT 6350 Homework 6
# Thomas Crawford

#install and load packages using install.packages & library fx
packagesNeeded = c("caret", "e1071", "mlbench", "corrplot", "pdp", "iml", 
                   "patchwork", "ggplot2", "gridExtra")
install.packages(packagesNeeded)

library(caret)
library(e1071)
library(mlbench)
library(corrplot)
library(pdp)
library(iml)
library(patchwork)
library(ggplot2)
library(gridExtra)

# ---------------------------------------------------------------------------
#Prepare training and test datasets
#Using the BostonHousing data, part of mlbench library
data(BostonHousing) #load the dataset
BH = BostonHousing #assign to new object for manipulation
str(BH) #check structure of the data
#medv is our outcome variable (y), all predictor variables are numeric or factors
#Check for any null values that need to be handled
sum(is.na(BH))

#Using set.seed() to reproduce results
set.seed(1234)

#Now using the createDataPartition fx to split into train and test sets
index = createDataPartition(BH$medv, p = 0.80, list = FALSE)
#Assign new index object to split BH into train & test
BHtrain = BH[index,]
BHtest = BH[-index,]
#Data has been split and is now prepared for modeling


# ---------------------------------------------------------------------------
#Gradient Boosted Tree
# ---------------------------------------------------------------------------
#Creating Gradient Boosted Tree model using caret's train() function
#Setting tuning parameters using trainControl() 
gbm_tune = trainControl(method = "repeatedcv", number = 10, repeats = 3)

#Now creating the model using train()
gbm_model = train(medv ~ ., data = BHtrain, method = "gbm", 
                  trControl = gbm_tune, verbose = FALSE)
#setting verbose to FALSE to stop model from printing all repeated runs
gbm_model$bestTune #see what tuning parameters were the best
#Best tune
# n.trees interaction.depth shrinkage n.minobsinnode
#9    150                 3       0.1             10

gbm_model #examine the model's results
summary(gbm_model) #examine our top predictors
#Our top 2 influential variables are lstat and rm. We will focus on these 2
#predictors when constructing our plots

# ----------------------------------------------
#Plotting our Gradient Boosted Model (PDP & ALE)
# ----------------------------------------------

#Partial Dependence Plots (PDP)
#Now we will be creating PDPs to examine the relationship lstat and rm with medv
gbm_var1 = partial(gbm_model, pred.var = "lstat", plot = TRUE)
gbm_var2 = partial(gbm_model, pred.var = "rm", plot = TRUE)
#With the PDPs created, now we use grid.arrange to plot the two together
grid.arrange(gbm_var1, gbm_var2, 
             top = "Partial Dependence Plots for Gradient Boosted Trees")

#Accumulated Local Effects (ALE) Plot
#Next we'll create an ALE plot from our Gradient Boosted Tree model
#Before we can create the plot, we first have to modify our original dataset to 
#remove the Y variable (medv) and create 2 separate objects: 1 for predictors, 
#the other for our outcome variable
BHdataX = BH[which(names(BH) != "medv")]
gbm_predictor = Predictor$new(gbm_model, data = BHdataX, y = BH$medv)

#Now we've created our X and Y variables, we can construct the ALE plot
gbm_ALE = FeatureEffects$new(gbm_predictor, method = "ale")
plot(gbm_ALE, features = c("lstat", "rm"))

# --------------------------------
#Evaluating our Model on Test Data
# --------------------------------

#We've evaluated the variable's of importance in our model, now we will test 
#and see how it performs on new test data using predict()
gbm_pred = predict(gbm_model, newdata = BHtest)

#Combine our observed and predicted medv values into a dataframe
gbm_df = data.frame(obs = BHtest$medv, pred = gbm_pred)

#Lastly use defaultSummary() to evaluate the model's performance
gbm_perf = defaultSummary(gbm_df)
gbm_perf
#Results:
#     RMSE  Rsquared       MAE 
#3.4316095 0.8146959 2.2186049
#Overall decent performance on our test data, I'd say an above average Rsquared
#value and RMSE. We will combine the defaultSummary of all our models at the end


# ---------------------------------------------------------------------------
#Random Forest
# ---------------------------------------------------------------------------
#Creating Random Forest model using caret's train() function
#Setting tuning parameters using trainControl()  and expand.grid
rf_tune = trainControl(method = "cv", number = 10)
rf_grid = expand.grid(.mtry = c(7)) 
#from our previous models, the best mtry from our train() model was 7

#Now creating the model using train()
rf_model = caret::train(medv ~ ., data = BHtrain, method = "rf", 
                  trControl = rf_tune, tuneGrid = rf_grid)
#Have to specify using caret's train() returns error message otherwise

rf_model #examine the model's results
#RMSE      Rsquared   MAE     
#3.209201  0.8873425  2.217208

varImp(rf_model) #examine our top predictors
#Our top 2 influential variables are lstat and rm again. We will focus on 
#these 2 predictors when constructing our plots

# -------------------------------------
#Plotting our Random Forest (PDP & ALE)
# -------------------------------------

#Partial Dependence Plots (PDP)
#Now we will be creating PDPs to examine the relationship lstat and rm with medv
rf_var1 = pdp::partial(rf_model, pred.var = "lstat", plot = TRUE)
rf_var2 = pdp::partial(rf_model, pred.var = "rm", plot = TRUE)
#Again having to specify use the pdp package partial() command
#With the PDPs created, now we use grid.arrange to plot the two together
grid.arrange(rf_var1, rf_var2, 
             top = "Partial Dependence Plots for Random Forest")

#Accumulated Local Effects (ALE) Plot
#Next we'll create an ALE plot from our Random Forest model
#Our predictor dataset with medv removed has already been created, and we're 
#calling it again in this function for rf_model
rf_predictor = Predictor$new(rf_model, data = BHdataX, y = BH$medv)

#Now we've created our X and Y variables, we can construct the ALE plot
rf_ALE = FeatureEffects$new(rf_predictor, method = "ale")
plot(rf_ALE, features = c("lstat", "rm"))

# --------------------------------
#Evaluating our Model on Test Data
# --------------------------------

#We've evaluated the variable's of importance in our model, now we will test 
#and see how it performs on new test data using predict()
rf_pred = predict(rf_model, newdata = BHtest)

#Combine our observed and predicted medv values into a dataframe
rf_df = data.frame(obs = BHtest$medv, pred = rf_pred)

#Lastly use defaultSummary() to evaluate the model's performance
rf_perf = defaultSummary(rf_df)
rf_perf
#Results:
#     RMSE  Rsquared       MAE 
#3.3016235 0.8267343 2.1201192
#Overall decent performance on our test data, model performed slightly better 
#than our Boosted Trees but overall could be better.
#We will combine the defaultSummary of all our models at the end


# ---------------------------------------------------------------------------
#Support Vector Machine - Polynomial
# ---------------------------------------------------------------------------
#Creating our Support Vector Machine model using caret's train() function
#Setting tuning parameters using trainControl() - repeated 10-fold crossvalidation
svm_tune = trainControl(method = "repeatedcv", number = 10, repeats = 3)

#Now creating the model using train()
svm_model = caret::train(medv ~ ., data = BHtrain, method = "svmPoly", 
                        trControl = svm_tune)
#MODEL TAKES A GOOD CHUNK OF TIME TO RUN
#Have to specify using caret's train() returns error message otherwise trying to
#use a different package's train() function

svm_model$bestTune #look at our model's optimal tuning parameter
#   degree scale   C
#26      3   0.1 0.5

varImp(svm_model) #examine our top predictors
#Unlike our Boosted Trees and Random Forest, our SVM has several influential
#predictors: nox, lstat, rm, & indus

# ----------------------------------------------------
#Plotting our Support Vector Machine (PDP, ICE, ALE, & Shapley)
# ----------------------------------------------------

#Partial Dependence Plots (PDP)
#Now we will be creating PDPs to examine the relationship of our predictor
#variables nox, lstat, rm, and indus with medv our outcome variable
svm_var1 = pdp::partial(svm_model, pred.var = "nox", plot = TRUE)
svm_var2 = pdp::partial(svm_model, pred.var = "lstat", plot = TRUE)
svm_var3 = pdp::partial(svm_model, pred.var = "rm", plot = TRUE)
svm_var4 = pdp::partial(svm_model, pred.var = "indus", plot = TRUE)
#Again having to specify use the pdp package partial() command
#With the PDPs created, now we use grid.arrange to plot the two together
grid.arrange(svm_var1, svm_var2, svm_var3, svm_var4, 
             top = "Partial Dependence Plots for Polynomial SVM")

#Individual Conditional Expectation (ICE) Plot
#With nox being a novel predictor variable not previously important in our 
#tree based models, we will examine it further using an ICE plot
svm_ice = pdp::partial(svm_model, pred.var = "nox", ice = TRUE, train = BHtrain)
#Now we plot the ICE curve
plotPartial(svm_ice, rug = TRUE, train = BHtrain, alpha = 0.5)

#Accumulated Local Effects (ALE) Plot
#Next we'll create an ALE plot from our Polynomial SVM model
#Our predictor dataset with medv removed has already been created, and we're 
#calling it again in this function with our svm_model
svm_predictor = Predictor$new(svm_model, data = BHdataX, y = BH$medv)

#Now we've created our X and Y variables, we can construct the ALE plot
svm_ALE = FeatureEffects$new(svm_predictor, method = "ale")
plot(svm_ALE, features = c("nox", "lstat", "rm", "indus"))


# --------------------------------
#Evaluating our Model on Test Data
# --------------------------------

#We've evaluated the variable's of importance in our model, now we will test 
#and see how it performs on new test data using predict()
svm_pred = predict(svm_model, newdata = BHtest)

#Combine our observed and predicted medv values into a dataframe
svm_df = data.frame(obs = BHtest$medv, pred = svm_pred)

#Lastly use defaultSummary() to evaluate the model's performance
svm_perf = defaultSummary(svm_df)
svm_perf
#Results:
#    RMSE Rsquared      MAE 
#2.895853 0.868048 1.954114
#SVM has the best Rsquared and RMSE values of our 3 models (as shown below),
#showing considerable improvement in predicting new data compared to our 
#tree based model

# --------------------------------
#Comparing our model performances
# --------------------------------

rbind("Gradient Boosted Trees" = gbm_perf,
      "Random Forest" = rf_perf,
      "Support Vector Machine - Polynomial" = svm_perf)
#Results:
#                                        RMSE  Rsquared      MAE
#Gradient Boosted Trees              3.431609 0.8146959 2.218605
#Random Forest                       3.301624 0.8267343 2.120119
#Support Vector Machine - Polynomial 2.895853 0.8680480 1.954114

#Our Polynomial SVM was by far our best model of the 3, showing a sizable
#increase in Rsquared and decrease in RMSE & MAE when compared to our tree 
#based models GBT and RF