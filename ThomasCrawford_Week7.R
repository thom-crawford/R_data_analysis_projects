# ----------------------
# QUMT 6350 Homework 7
# Thomas Crawford

#install and load packages using install.packages & library fx
packagesNeeded = c("caret", "e1071", "randomForest")
install.packages(packagesNeeded)

library(caret)
library(e1071)
library(randomForest)

# ---------------------------------------------------------------------------
#Prepare training and test datasets
#Using the iris dataset, part of R's base library
data("iris") #load the dataset
irisData = iris #assign to new object for manipulation
str(irisData) #check structure of the data
#The flower species is our y/outcome variable of interest 

#Using set.seed() to reproduce results
set.seed(1234)

#Now using the createDataPartition fx to split into train and test sets
index = createDataPartition(irisData$Species, p = 0.75, list = FALSE)
#Assign new index object to split irisData into train & test
irisTrain = irisData[index,]
irisTest = irisData[-index,]
#Data has been split and is now prepared for modeling

# -----------------------------------------------------------------------------
#Random Forest using randomForest()
# -----------------------------------------------------------------------------
#PART A

#We start by creating our model using the randomForest() function, leaving our 
#ntrees and mtry values at their default/non-tuned values
#Also setting importance = TRUE so we can see how our predictor variables influence
#our Outcome variable - Species

irisModel1 = randomForest(Species ~ ., data = irisTrain, importance = TRUE)

varImp(irisModel1) #see what variables are the most important
varImpPlot(irisModel1) #create a plot showing our important predictors
#Looks like Petal.Length & Petal.Width are the most important predictors

#Now making predictions using our model on the test dataset
irisPred = predict(irisModel1, newdata = irisTest, type = "response")

#Creating a confusion matrix to evaluate our model performance on the test data
irisConfusion = confusionMatrix(data = irisPred, reference = irisTest$Species)
irisConfusion
#Model performed pretty well on our test data, with accuracy of 0.8889:

#Confusion Matrix and Statistics
#            Reference
#Prediction   setosa versicolor virginica
#  setosa         12          0         0
#  versicolor      0         11         3
#  virginica       0          1         9

#Overall Statistics
#Accuracy : 0.8889          
#P-Value [Acc > NIR] : 6.677e-12       
#Kappa : 0.8333 
#Statistics by Class:
#                     Class: setosa Class: versicolor Class: virginica
#Sensitivity                 1.0000            0.9167           0.7500
#Specificity                 1.0000            0.8750           0.9583

#Create new data.frame to house overall accuracy from confusionMatrix for 
#comparison at the end
iris_cm1 = data.frame(Accuracy = irisConfusion$overall["Accuracy"],
                     row.names = "irisModel1")

# --------------------------------------
#Part B

#Since evaluating the full model, now we'll trim the number of predictor variables
#to remove the 2 least important predictors (as was shown earlier in our varImp
#and varImpPlot commands)

#Again using the randomForest() function, with Sepal.Length & Sepal.Width removed
#as predictors

irisModel2 = randomForest(Species ~ Petal.Length + Petal.Width, data = irisTrain,
                          importance = TRUE)
#With our updated model, we'll now create predictions and measure the model's 
#performance on the test data using a confusionMatrix as we did above
irisPred2 = predict(irisModel2, newdata = irisTest, type = "response")
irisConfusion2 = confusionMatrix(data = irisPred2, reference = irisTest$Species)
irisConfusion2
#Removing Sepal.Length and Sepal.width increased our model's performance from 
#0.8889 to 0.9444. Our updated model correctly identifed more of the versicolor
#and virginica species than the prior one

#Confusion Matrix and Statistics
#            Reference
#Prediction   setosa versicolor virginica
#  setosa         12          0         0
#  versicolor      0         11         1
#  virginica       0          1        11

#Overall Statistics
#Accuracy : 0.9444          
#P-Value [Acc > NIR] : 1.728e-14       
#Kappa : 0.9167
#Statistics by Class:
#                     Class: setosa Class: versicolor Class: virginica
#Sensitivity                 1.0000            0.9167           0.9167
#Specificity                 1.0000            0.9583           0.9583

#Create new data.frame to house overall accuracy from confusionMatrix for 
#comparison at the end
iris_cm2 = data.frame(Accuracy = irisConfusion2$overall["Accuracy"],
                      row.names = "irisModel2")

# -----------------------------------------------------------------------------
#Part C: Random Forest using train()
# -----------------------------------------------------------------------------

#We've created 2 different models using the specific randomForest command from 
#the randomForest package, now we will compared those models to a new randomForest
#model created using the train() function from the caret package

#We are setting tuning parameters for our model to use 10-fold cross validation
#and using all of our predictor variables

irisModel3 = train(Species ~ ., data = irisTrain, method = "rf",
                   trControl = trainControl(method = "cv", number = 10))

varImp(irisModel3) #see what our most important predictors are in the model
#varImpPlot(irisModel3) #Does not work because output from train() is wrong class
#plot function only works for randomForest model if the class = randomForest
class(irisModel3) #shows the object class is "train.formula" 
#Our most important variables are the same as above: Petal.Width & Petal.Length

#Next we'll create predictions using our model and the test data set
irisPred3 = predict(irisModel3, newdata = irisTest, type = "raw")

#As before, we will create a confusionMatrix to evaluate our model's performance
irisConfusion3 = confusionMatrix(data = irisPred3, reference = irisTest$Species)
irisConfusion3
#Our train() model performed very similar to our 1st randomForest model with all
#predictors included with an overall accuracy of 0.8889 
#Confusion Matrix and Statistics
#Reference
#Prediction   setosa versicolor virginica
#  setosa         12          0         0
#  versicolor      0         11         3
#  virginica       0          1         9

#Overall Statistics
#Accuracy : 0.8889          
#P-Value [Acc > NIR] : 6.677e-12       
#Kappa : 0.8333          

#Statistics by Class:
#                     Class: setosa Class: versicolor Class: virginica
#Sensitivity                 1.0000            0.9167           0.7500
#Specificity                 1.0000            0.8750           0.9583

#Create new data.frame to house overall accuracy from confusionMatrix for 
#comparison at the end
iris_cm3 = data.frame(Accuracy = irisConfusion3$overall["Accuracy"],
                      row.names = "irisModel3")

# -------------------------------------------------------------------------
#Comparing Model Performance

rbind(iris_cm1, iris_cm2, iris_cm3)
#            Accuracy    #tried to pull Sensitivity and Specificity but was unable
#irisModel1 0.8888889
#irisModel2 0.9444444
#irisModel3 0.8888889

#Our model with the 2 least important predictors removed (irisModel2) performed
#the best on the data all 3 models correctly identifed all setosa species, but
#irisModel2 correctly identifed more of the versicolor and virginica species
#than irisModel1 and irisModel3

#----------------------------------------
#irisModel1
#            Reference
#Prediction   setosa versicolor virginica
#  setosa         12          0         0
#  versicolor      0         11         3
#  virginica       0          1         9
#                     Class: setosa Class: versicolor Class: virginica
#Sensitivity                 1.0000            0.9167           0.7500
#Specificity                 1.0000            0.8750           0.9583

#----------------------------------------
#irisModel2
#            Reference
#Prediction   setosa versicolor virginica
#  setosa         12          0         0
#  versicolor      0         11         1
#  virginica       0          1        11
#                     Class: setosa Class: versicolor Class: virginica
#Sensitivity                 1.0000            0.9167           0.9167
#Specificity                 1.0000            0.9583           0.9583

#----------------------------------------
#irisModel3
#            Reference
#Prediction   setosa versicolor virginica
#  setosa         12          0         0
#  versicolor      0         11         3
#  virginica       0          1         9
#                     Class: setosa Class: versicolor Class: virginica
#Sensitivity                 1.0000            0.9167           0.7500
#Specificity                 1.0000            0.8750           0.9583
