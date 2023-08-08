#install and load packages using install.packages & library fx
packagesNeeded = c("caret", "e1071", "mlbench", "corrplot", "pls",
                   "glmnet", "earth", "kernlab")
install.packages(packagesNeeded)

library(caret)
library(e1071)
library(mlbench)
library(corrplot)
library(pls)
library(glmnet)
library(earth)
library(kernlab)

# -----------------------
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

# -----------------------
#Linear Regression
# -----------------------

#Using the lm function to create linear model using medv as our y-var
BH_lm = lm(medv ~., data = BHtrain)
summary(BH_lm) #to view how our model performed on the training data
#Next taking the generated model and running against the test data
lm_predict = predict(BH_lm, BHtest)
#Now create dataframe to house our observed and predicted medv
lm_df = data.frame(obs = BHtest$medv, pred = lm_predict)
#lastly, use the defaultSummary function to estimate the model's performance 
#on the test data
lm_performance = defaultSummary(lm_df)
lm_performance
#RMSE = 6.01717, Rsquared = 0.64289, MAE = 3.84608

# -----------------------
# Partial Least Squares 
# -----------------------

#Using the plsr() fx from pls library to run 2 different models
#1st variation will be using SIMPLS and specifying 10-fold cross-validation
BH_simpls = plsr(medv ~ ., data=BHtrain, method = "simpls",
                 validation = "CV")
summary(BH_simpls) #Use summary to observe model performance
#SIMPLS model says that only 10 components are need to explain 100% of the 
#variance as shown under the section labelled "Training"
#Next we will make predictions using the BH_simpls model on test data 
#using 10 components instead of all 13
simpls_predict = predict(BH_simpls, BHtest, ncomp = 10) #ncomp() to specify # of components
#Now we will organized the observed and predicted results into a data frame
simpls_df = data.frame(obs = BHtest$medv, pred = simpls_predict[,,])
#Use defaultSummary() to observe how our model performed on the test data
simpls_performance = defaultSummary(simpls_df)
simpls_performance
#RMSE = 5.86866, Rsqured = 0.66694, MAE = 3.93121

#Our 2nd variation will be using the kernel algorithm and specifying 10-fold CV
BH_kernpls = plsr(medv ~., data = BHtrain, method = "kernelpls",
                  validation = "CV")
summary(BH_kernpls) #using summary to obverse performance and min # of components
#Summary of our kernel pls model says the minimum components to explain 100% of
#the variance is 10 components
#Now we make prediction using the model and the test data
kernpls_predict = predict(BH_kernpls, BHtest, ncomp = 10) #using ncomp() to specify
#Using data.frame() to house our observed and predicted medv values
kernpls_df = data.frame(obs = BHtest$medv, pred = kernpls_predict[,,])
#Use defaultSummary() on our data frame to measure the performance on test data
kernpls_performance = defaultSummary(kernpls_df)
kernpls_performance
#RMSE = 5.86866, Rsquared = 0.66694, MAE = 3.93121

# ---------------------------
# Penalized Regression Models
# ---------------------------

#Going to focus on running 2 different penalized regression models: Ridged & Lasso
#First we will create and measure the Ridge regression model

#Ridge Regression:
#glmnet() requires x and y values be specified
x = model.matrix(medv ~., data=BHtrain)[,-1] #create our x values/predictors
y = BHtrain$medv #create our y variable with the output
#Now we will specify a values for lambda to be used in the model
set.seed(123)
cv = cv.glmnet(x,y, alpha = 0) #using the cv.glmnet to optimize lambda by cross-validation
ridge_lambda = cv$lambda.min #selecting our optimal value of lambda for the model
#Use glmnet() to create our ridge regression model
BH_ridge = glmnet(x, y, alpha = 0, lambda = ridge_lambda)
#Now to specify our x test variables to use to make our predictions
xtest = model.matrix(medv ~., data=BHtest)[,-1]
#Use predict() and xtest variable to make predictions using our model
ridge_predict = predict(BH_ridge, xtest)
#Consolidate our observed and predicted values into a data frame
ridge_df = data.frame(obs = BHtest$medv, pred = ridge_predict[,])
#Use defaultSummary() to measure the model's performance on test data
ridge_performance = defaultSummary(ridge_df)
ridge_performance
#RMSE = 6.19283, Rsquared = 0.62279, MAE = 3.94069

#Lasso Regression:
#Using the same steps and x/y objects from ridge regression, we can fit
#the lasso regression to the training data
#Specifying value of lambda for the lasso regression using cv.glmnet()
set.seed(234)
cv2 = cv.glmnet(x, y, alpha = 1)
lasso_lambda = cv2$lambda.min #capturing our optimized lambda value
#Using glmnet() with alpha=1 to create lasso model instead of ridge model
BH_lasso = glmnet(x, y, alpha = 1, lambda = lasso_lambda)
#Use predict() to make predictions on the test data (xtest)
lasso_predict = predict(BH_lasso, xtest)
#Combine observed and predicted medv into a data frame
lasso_df = data.frame(obs = BHtest$medv, pred = lasso_predict[,])
#Lastly, use defaultSummary() to evaluate model performance on test data
lasso_performance = defaultSummary(lasso_df)
lasso_performance
#RMSE = 6.02624, Rsquared = 0.64189, MAE = 3.85099

# -----------------------------------------------
# Multivariate Adaptive Regression Splines (MARS)
# -----------------------------------------------

#Create a MARS model using the earth() function from the earth package
BH_mars = earth(medv ~ ., data = BHtrain, nfold = 10) #using nfold to specify cross validation
summary(BH_mars) #observe results from the model
#Use predict() to test model on our test dataset BHtest
mars_predict = predict(BH_mars, BHtest)
#Combine observed and predicted medv into a single data frame
mars_df = data.frame(obs = BHtest$medv, pred = mars_predict[,])
#Meausre the model's performance on the test data using defaultSummary()
mars_performance = defaultSummary(mars_df)
mars_performance
#RMSE = 4.16464, Rsquared = 0.83088, MAE = 2.52865

# ------------------------
# Support Vector Machines
# ------------------------

#Creating 2 Support Vector Machine Models using the train() function
#Both models we will use 10-fold cross-validation, creating a data object to handle
t_control = trainControl(method = "cv", number = 10)

#1st model: Radial SVM
BH_svmrad = train(medv ~ ., data = BHtrain, method = "svmRadial",
                  trControl = t_control)
BH_svmrad #examine the model details
#Using predict() to test our model on the BHtest data
svmrad_predict = predict(BH_svmrad, BHtest)
#Combine the observed and predicted medv values into data frame
svmrad_df = data.frame(obs = BHtest$medv, pred = svmrad_predict)
#use the defaultSummary() to examine the model's performance on test data
svmrad_performance = defaultSummary(svmrad_df)
svmrad_performance
#RMSE = 5.85579, Rsquared = 0.69180, MAE = 2.97493

#2nd Model: Polynomial SVM
BH_svmpoly = train(medv ~ ., data = BHtrain, method = "svmPoly",
                   trControl = t_control)
BH_svmpoly #examine model details
#Use predict() to test model on the BHtest data
svmpoly_predict = predict(BH_svmpoly, BHtest)
#Combine our observed and predicted values into a data frame
svmpoly_df = data.frame(obs = BHtest$medv, pred = svmpoly_predict)
#Use the defaultSummary() to evaluates the performance on test data
svmpoly_performance = defaultSummary(svmpoly_df)
svmpoly_performance
#RMSE = 5.44785, Rsquared = 0.70682, MAE = 2.86897

# -----------------------
# K-Nearest Neighbor
# -----------------------

#Again using the train() command to create our KNN model
BH_knn = train(medv ~ ., data = BHtrain, method = "knn", 
               trControl = t_control) #using 10-fold cross-validation
BH_knn #Examine our KNN model - best K-value = 5
#Making predictions on the test data using predict()
knn_predict = predict(BH_knn, BHtest)
#Create data frame to house our observed and predicted values
knn_df = data.frame(obs = BHtest$medv, pred = knn_predict)
#Use defaultSummary() to measure model performance on test data
knn_performance = defaultSummary(knn_df)
knn_performance
#RMSE = 7.41143, Rsquared = 0.45953, MAE = 4.88929

# -----------------------
# Hyperparameter Tuning
# -----------------------
#We will be conducting 3 hyperparameter tunings on the SVM Models

#The first hyperparameter tuning will be changing the default grid search
#in the trainControl() function to a random search along with cross-validation
#to create the SVM Polynomial model
svmpoly_random = train(medv ~ ., data = BHtrain, method = "svmPoly",
                       trControl = trainControl(method = "cv",
                                                number = 10,
                                                search = "random"))
svmpoly_random
#With our tuned model, we will create predictions on the test data
svmpolyRand_predict = predict(svmpoly_random, BHtest)
#Create data frame combining observed and predicted values
poly_rand_df = data.frame(obs = BHtest$medv, pred = svmpolyRand_predict)
#Use defaultSummary() to evaluate test performance
poly_rand_performance = defaultSummary(poly_rand_df)
poly_rand_performance
#RMSE = 6.48388, Rsquared = 0.5891, MAE = 3.88202

#Our 2nd hyperparameter tuning will be a guided grid search in the trainControl()
#Call SVM Poly function from earlier to obtain best parameters
BH_svmpoly$bestTune
#Our best parameters are : Degree = 3, Scale = 0.1, & C = 0.5
#Taking our best parameters, and creating data-frame for tuneGrid in train() to use
svmpoly_grid = data.frame(.C = c(0.5), .degree = c(3), .scale = c(0.1))
#Now with our guided search, we create the SVM Polynomial model
svmpoly_guided = train(medv ~ ., data = BHtrain, method = "svmPoly",
                       tuneGrid = svmpoly_grid,
                       trControl = trainControl(method = "cv",
                                                number = 10,
                                                search = "grid"))
svmpoly_guided #examine our model
#Taking our tuned guided model, now will make predictions on test data
svmpolyGuided_predict = predict(svmpoly_guided, BHtest)
#Create dataframe combining our observed and predicted values
poly_guided_df = data.frame(obs = BHtest$medv, pred = svmpolyGuided_predict)
#Use defaultSummary() to measure our model's performance
poly_guided_performance = defaultSummary(poly_guided_df)
poly_guided_performance
#RMSE = 5.44786, Rsquared = 0.70682, MAE = 2.86897

#For our 3rd hyperparameter tuning, we will use the same guided search from above,
#but also change the method and number of cross-validation performed with the model
svmpoly_guided_rcv = train(medv ~ ., data = BHtrain, method = "svmPoly",
                       tuneGrid = svmpoly_grid,
                       trControl = trainControl(method = "repeatedcv",
                                                number = 10,
                                                repeats = 3,
                                                search = "grid"))
svmpoly_guided_rcv #examine our model
#Using our tuned model, will now apply to our BHtest dataset
poly_guided_rcv_predict = predict(svmpoly_guided_rcv, BHtest)
#Combine our observed and predicted medv values
poly_guided_rcv_df = data.frame(obs = BHtest$medv, pred = poly_guided_rcv_predict)
#Use defaultSummary() to evaluate our model's performance on test data
poly_guided_rcv_performance = defaultSummary(poly_guided_rcv_df)
poly_guided_rcv_performance
#RMSE = 5.44785, Rsquared = 0.70682, MAE = 2.86897

#lastly using Rbind() to combine all of our model's performance
rbind(lm_performance,simpls_performance, kernpls_performance, ridge_performance,
      lasso_performance, mars_performance, svmrad_performance, svmpoly_performance,
      knn_performance, poly_rand_performance, poly_guided_performance, poly_guided_rcv_performance)
#Based on our generated models, the Multivariate Adaptive Regression Splines model
#performed the best on our test data
