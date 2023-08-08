#1) Download packages using install.packages
install.packages("caret")
install.packages("e1071")
install.packages("corrplot")
install.packages("mlbench")

#Use library to load the packages for use
library(caret)
library(e1071)
library(corrplot)
library(mlbench)

#2) Loading the Glass dataset and assigning to new object "gdata"
data(Glass)
gdata = Glass

#3) Using set.seed() command to allow exact results to be reproduced
set.seed(114)

#4a) Creating train and test sets using sample command
index = sample(nrow(gdata), nrow(gdata)*0.8) #takes gdata and creates sample from 80% of it
gtrain = gdata[index,]
gtest = gdata[-index,]

#4b) Doing the same as above but using createDataPartition() from caret package
index2 = createDataPartition(gdata$Type, p = 0.8, list = FALSE)
gtrains = gdata[index2,]
gtests = gdata[-index2,]

#5) 
#Using View() to examine the entire dataset
View(gdata)
#Using head() to look at the first 6 rows of the dataset
head(gdata)
#Using str() to examine the structure of the data
str(gdata) #10 columns in total, 1-9 are numerical and 10 is Type which is a factor

#6) Saving Ca, Si, and Type columns are vectors
Calcium = gdata["Ca"]
Silicon = gdata["Si"]
Type = gdata["Type"]

#7) Create new dataset with Ca, Si, and Type columns removed
gdata2 = gdata[, -c(5,7,10)]
str(gdata2) #use to verify new set doesn't include Ca, Si, and Type

#8a) Examining the skewness of gdata
#Using apply() to create a loop and apply the skewness function to all columns
#in the dataset
apply(gdata2, 2, skewness)

#8b) Examine the skewness of K using density() to measure the skewness and 
#plot() to graph it
plot(density(gdata2$K))

#9) Using preProcess() to center and scale data, reducing skewness
preProcValues = preProcess(gtrain, method = c("center", "scale"))

#10) Using the adjusted values from #9 and predict() to center & scale gtrain and gtest
gtrain_cs = predict(preProcValues, gtrain)
gtest_cs = predict(preProcValues, gtest)

#11a) Using preProcess() again on gtrain using BoxCox & YeoJohnson methods
preProcValuesBC = preProcess(gtrain, method = "BoxCox")
preProcValuesYJ = preProcess(gtrain, method = "YeoJohnson")
#Use the predict() to apply BoxCox and YeoJohnson processed values to gtrain
gtrainBC = predict(preProcValuesBC, gtrain)
gtrainYJ = predict(preProcValuesYJ, gtrain)

#11b) Check skewness of both gtrainBC and gtrainYJ with Type column removed
#using apply() look again to apply skewness to all columns except Type
apply(gtrainBC[,-10], 2, skewness)
#Skewness not well reduced by BoxCox method
apply(gtrainYJ[,-10], 2, skewness)
#Skewness significantly reduced using YeoJohnson

#11c) Plot the skewness of K from gtrainYJ using denisty() and plot()
plot(density(gtrainYJ$K))

#12) Creating correlation matrix on gdata2 to examine any linearity
correlations = cor(gdata2)
correlations #examine the correlation matrix

#13) Taking the correlations object from 12 and creating plot showing the correlations
corrplot(correlations)

#14) Creating correlation matrix from gdata with Type column removed
correlations_noType = cor(gdata[,-10])
correlations_noType #examine the correlation matrix w/o Type column

#15) Check for missing data in the original gdata dataset
is.na(gdata) #check for any missing data in entire set
sum(is.na(gdata)) #total count of missing data

#16a) Run linear regression on preprocessed dataset gtrainYJ
#Using K as our Y and Mg & Ba as our X, along with Mg:Ba interaction
lmresult = lm(K ~ Mg*Ba, data = gtrainYJ) #Using "*" to signify our predictors - Mg + Ba + Mg:Ba

#16b) using summary() to view results of the linear regression
summary(lmresult)
#Mg and Mg:Ba interaction are statistically significant to K
