
# Individual Study Research 


library(e1071)
library(caret)
library(tseries)
library(dataseries)
library(ISLR)
library(forecast)
library(randomForest)
library(tidyverse)
library(neuralnet)
library(MASS)
library(Metrics)


set.seed(6822651)
# Load data
data = read.csv("FB.csv")

# we apply the natural logarithmic to stabilize or normalize the Data
data$Open = log(data$Open)
data$High = log(data$High)
data$Low = log(data$Low)
data$Close = log(data$Close)
data$Adj.Close = log(data$Adj.Close)
data$Volume = log(data$Volume)
head(data)
tail(data)

#Data Visualization
# Daily Seasonality (frequency set to 1 for daily data)
Close.FB <- ts(data[,5], frequency = 1)

#Delimit train range
Close_train <- window(Close.FB, end = c(1208,1))
Close_test <- window(Close.FB, start = c(1209,1))

#Training and testing data visualization
plot(Close.FB, main="Facebook-Close", ylab="Close", xlab="Days")
lines(Close_train, col = "blue")
lines(Close_test, col = "green")
legend(y=5.7, x=15, legend = c("Training", "Testing"), col = c("blue", "green"), lty = 1)

#Data reprocessing 
str(data) # "Date" is a Factor
# Convert "Date" as numeric
data$Date <- as.numeric(as.factor(data$Date))

# Creating training and test data
data.train <- data[1:1208, ]
data.test <- data[1209:1510, ]


#Fit the SUPPORT VECTOR REGRESSOR
#Find the best parameter of the SVM model
set.seed(6822651)
tune.out=tune(svm, Close ~., data=data.train, kernel= "radial",
              ranges =list(cost=c(0.1 ,1, 5, 10),
                           gamma=c(0.01, 0.1, 1, 5, 10, 100)))

summary(tune.out) # best cost=10 and gamma=0.01

# Fit the SVM-Regressor
svm.model.regressor <- svm(Close~., data=data.train, cost=10, gamma = 0.01,
                           kernel ="radial", type="eps-regression")
summary(svm.model.regressor)

#Close Price prediction using SVM Model
svm.pred.test = predict(svm.model.regressor, data.test)
head(svm.pred.test)
svm.pred.train = predict(svm.model.regressor, data.train)
head(svm.pred.train)

#Calculate the RMSE using SVM Model
rmse.svm <- rmse(svm.pred.test, data.test$Close)
rmse.svm 

#Plot the prediction of the SVM for Regression 
plot(data$Close, type = "l", xlab = "Days", ylab="Close", main="Observed, Fitted and predicted Values using svm")
lines(x = c(1209:1510), svm.pred.test, col="green")
lines(x = c(1:1208), svm.pred.train, col="blue")
legend(y = 5.75, x = 0, legend = c("Obs", "pred.train", "pred.test"), col = c("black", "blue","green"), lwd = 1)

#data denormalization
# Results
prediction <- exp(svm.pred.test)
actual <- exp(data.test$Close)
results1 <- data.frame( actual = actual, Prediction = prediction)
Days <- data.test$Date 
results1 <- cbind(Days, results1)
head(results1)
tail(results1)

#Fit the RandomForet Model
# Find the best mtry
set.seed(6822651)
bestmtry <- tuneRF(data.train[,-5], data.train[, 5], ntreeTry=50, 
                   stepFactor=2, improve=0.05) # The best mtry is equal to 6

#Fit the RF Model
RF.model <- randomForest(Close ~., data=data.train, 
                        mtry = 6, importance =TRUE)
summary(RF.model)


##Close Price prediction using RandomForest Model
RF.pred.test = predict(RF.model, data.test)
head(RF.pred.test)
RF.pred.train = predict(RF.model, data.train)
head(RF.pred.train)

#Calculate the MSE using RandomForest Model
rmse.RF <- rmse(RF.pred.test, data.test$Close)
rmse.RF

#Plot the prediction of the RF 
plot(data$Close, type = "l", xlab = "Days", ylab="Close", main="Observed, Fitted and predicted Values using RF")
lines(x = c(1209:1510), RF.pred.test, col="green")
lines(x = c(1:1208), RF.pred.train, col="blue")
legend(y = 5.75, x = 5, legend = c("Obs", "pred.test", "pred.train"), col = c("black", "green","blue"), lwd = 1)

#data denormalization
# Results
prediction <- exp(RF.pred.test)
actual <- exp(data.test$Close)
results2 <- data.frame( actual = actual, Prediction = prediction)
Days <- data.test$Date 
results2 <- cbind(Days, results2)
head(results2)
tail(results2)

#Fit the Artificial Neural Network(ANN) model
#Find the best Hidden with smallest rmse

# hidden = 1
set.seed(6822651)
nn1.model <- neuralnet(Close~., data=data.train, hidden = 1, err.fct = "sse",  
                      threshold = 0.04, act.fct = "tanh", stepmax = 1e7, linear.output = T)
nn1.pred <- predict(nn1.model, data.test)
rmse.nn1 <- rmse(nn1.pred, data.test$Close)
rmse.nn1


# hidden = 4
set.seed(6822651)
nn4.model <- neuralnet(Close~., data=data.train, hidden = 4, err.fct = "sse",  
                       threshold = 0.04, act.fct = "tanh", stepmax = 1e7, linear.output = T)
nn4.pred <- predict(nn4.model, data.test)
rmse.nn4 <- rmse(nn4.pred, data.test$Close)
rmse.nn4

# hidden = 5
set.seed(6822651)
nn5.model <- neuralnet(Close~., data=data.train, hidden = 5, err.fct = "sse",  
                       threshold = 0.04, act.fct = "tanh", stepmax = 1e7, linear.output = T)
nn5.pred <- predict(nn5.model, data.test)
rmse.nn5 <- rmse(nn5.pred, data.test$Close)
rmse.nn5

# hidden = 6
set.seed(6822651)
nn6.model <- neuralnet(Close~., data=data.train, hidden = 6, err.fct = "sse",  
                       threshold = 0.04, act.fct = "tanh", stepmax = 1e7, linear.output = T)
nn6.pred <- predict(nn6.model, data.test)
rmse.nn6 <- rmse(nn6.pred, data.test$Close)
rmse.nn6

# hidden = 7
set.seed(6822651)
nn7.model <- neuralnet(Close~., data=data.train, hidden = 7, err.fct = "sse",  
                       threshold = 0.04, act.fct = "tanh", stepmax = 1e7, linear.output = T)
nn7.pred <- predict(nn7.model, data.test)
rmse.nn7 <- rmse(nn7.pred, data.test$Close)
rmse.nn7

# hidden = 10
set.seed(6822651)
nn10.model <- neuralnet(Close~., data=data.train, hidden = 10, err.fct = "sse",  
                        threshold = 0.04, act.fct = "tanh", stepmax = 1e7, linear.output = T)
nn10.pred <- predict(nn10.model, data.test)
rmse.nn10 <- rmse(nn10.pred, data.test$Close)
rmse.nn10

# hidden = 15
set.seed(6822651)
nn15.model = neuralnet(Close~., data=data.train, hidden = 15, err.fct = "sse",  
                       threshold = 0.04, act.fct = "tanh", stepmax = 1e7, linear.output = T)
nn15.pred <- predict(nn15.model, data.test)
rmse.nn15 <- rmse(nn15.pred, data.test$Close)
rmse.nn15

# The best Hidden = 5 with rmse = 0.02248372

nn.model <- nn5.model
nn.pred.test <- nn5.pred
nn.pred.train <- predict(nn.model, data.train)
rmse.nn <- rmse.nn5
rmse.nn

summary(nn.model)

plot(nn.model, rep = "best")

#Plot the prediction of the ANN 
plot(data$Close, xlab = "Days", ylab="Close", type= "l", main="Observed, Fitted and predicted Values using ANN")
lines(x = c(1209:1510), nn.pred.test, col="green")
lines(x = c(1:1208), nn.pred.train, col="blue")
legend(y = 5.75, x = 5, legend = c("Obs", "pred.test", "pred.train"), col = c("black", "green","blue"), lwd = 1)

#data denormalization
# Results
prediction <- exp(nn.pred.test)
actual <- exp(data.test$Close)
results3 <- data.frame( actual = actual, Prediction = prediction)
Days <- data.test$Date 
results3 <- cbind(Days, results3)
head(results3)
tail(results3)


# Using the PCA processing data
# Preparing data to PCA
# Correlation between different Variables
library(corrplot)
cor_vals <- round(cor(data),5)
corrplot(cor_vals, type = "upper", order = "hclust", tl.col = "black", tl.srt = 45)

# PCA model
pca <- prcomp(data.train[, -5], center = T, scale. = T ) # PCA need only independents variables
summary(pca) 

#variance
pr.var <- (pca$sdev)^2

#% of variamce
prop.varex <- pr.var / sum(pr.var)

# plot
plot(prop.varex, xlab = "Principal Component",
     ylab = "Proportion of variance Explained",
     type = "b")

#Scree Plot
plot(cumsum(prop.varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")
##By observing the output of the PCA, we can see that the first two PC's explain 99% of the variability in the data.
# We can only use PC1, PCA2 and PC3

# Add a training set with principal components
train.data <- predict(pca, data.train)
train.data <- data.frame(train.data, data.train[5])
head(train.data)
test.data <- predict(pca, data.test)
test.data <- data.frame(test.data, data.test[5])
head(test.data)

#We are interested in first 3 PCAs
# We can only use PC1, PC2 and PC3
new.data.train <- train.data[, c(1:3,7)]
new.data.test <- test.data[, c(1:3,7)]


# Fit the SVM-Regressor model with PCA
# Find the best Parameters
set.seed(6822651)
tune.out=tune(svm, Close ~ PC1+PC2+PC3, data=new.data.train, kernel ="radial",
              ranges =list(cost=c(0.1 ,1, 5, 10),
                           gamma=c(0.01, 0.1, 1, 5, 10, 100)))
summary(tune.out) # The best Cost=10 and gamma=0.01

#Fit the model
svm.pca <- svm(Close ~ PC1+PC2+PC3, data=new.data.train, cost=10, gamma=0.01,
               kernel ="radial", type="eps-regression")
summary(svm.pca)

#Close Price prediction using SVM Model with PCA
svm.pca.pred <- predict(svm.pca, test.data)
head(svm.pca.pred)
svm.pca.train <- predict(svm.pca, train.data)
head(svm.pca.train)

# Calculating of RMSE
rmse.svm.pca <- rmse(svm.pca.pred, new.data.test$Close)
rmse.svm.pca 

#Plot the prediction of the SVM for Regression wiht pca
plot(data$Close, type = "l", xlab = "Days", ylab="Close", main="Observed, Fitted and predicted Values using PCA-SVM")
lines(x = c(1209:1510), svm.pca.pred, col="green")
lines(x = c(1:1208), svm.pca.train, col="blue")
legend(y = 5.75, x = 5, legend = c("Obs", "pred.pca.test", "pred.pca.train"), col = c("black", "green","blue"), lwd = 1)

#data denormalization
# Results
prediction <- exp(svm.pca.pred)
actual <- exp(data.test$Close)
results1 <- data.frame( actual = actual, Prediction = prediction)
Days <- data.test$Date 
results1 <- cbind(Days, results1)
head(results1)
tail(results1)

#Fit the RandomForet Model with PCA
# Find the best mtry
set.seed(6822651)
bestmtry <- tuneRF(new.data.train[,-4], new.data.train[,4], ntreeTry=50, 
                   stepFactor=2, improve=0.05) # The best mtry is equal to 3

# Fit the model
RF.pca <- randomForest(Close ~ PC1+PC2+PC3, data=new.data.train, 
                       mtry = 3, importance =TRUE)
summary(RF.pca)

#Close Price prediction using RandomForest Model
RF.pca.pred = predict(RF.pca, new.data.test)
head(RF.pca.pred)
RF.pca.train = predict(RF.pca, new.data.train)
head(RF.pca.train)

#Calculate the MSE using RandomForest Model
rmse.RF.pca <- rmse(RF.pca.pred, new.data.test$Close)
rmse.RF.pca

#Plot the prediction of the RF wiht pca
plot(data$Close, type = "l", xlab = "Days", ylab="Close", main="Observed, Fitted and predicted Values using PCA-RF")
lines(x = c(1209:1510), RF.pca.pred, col="green")
lines(x = c(1:1208), RF.pca.train, col="blue")
legend(y = 5.75, x = 5, legend = c("Obs", "pred.pca.test", "pred.pca.train"), col = c("black", "green","blue"), lwd = 1)

#data denormalization
# Results
prediction <- exp(RF.pca.pred)
actual <- exp(data.test$Close)
results2 <- data.frame( actual = actual, Prediction = prediction)
Days <- data.test$Date 
results1 <- cbind(Days, results1)
head(results2)
tail(results2)

#Fit the Artificial Neural Network(ANN) model with PCA
#Find the best Hidden with smallest rmse for Hidden between 1 and 20

# hidden = 1
set.seed(6822651)
nn1.pca = neuralnet(Close ~ PC1+PC2+PC3, data=new.data.train, hidden = 1, err.fct = "sse",  
                    threshold = 0.04, act.fct = "tanh", stepmax = 1e7, linear.output = T)
nn1.pred.pca <- predict(nn1.pca, new.data.test)
rmse.nn1.pca <- rmse(nn1.pred.pca, new.data.test$Close)
rmse.nn1.pca

# hidden = 2
set.seed(6822651)
nn2.pca = neuralnet(Close ~ PC1+PC2+PC3, data=new.data.train, hidden = 2, err.fct = "sse",  
                    threshold = 0.04, act.fct = "tanh", stepmax = 1e7, linear.output = T)
nn2.pred.pca <- predict(nn2.pca, new.data.test)
rmse.nn2.pca <- rmse(nn2.pred.pca, new.data.test$Close)
rmse.nn2.pca

# hidden = 3
set.seed(6822651)
nn3.pca = neuralnet(Close ~ PC1+PC2+PC3, data=new.data.train, hidden = 3, err.fct = "sse",  
                    threshold = 0.04, act.fct = "tanh", stepmax = 1e5, linear.output = T)
nn3.pred.pca <- predict(nn3.pca, new.data.test)
rmse.nn3.pca <- rmse(nn3.pred.pca, new.data.test$Close)
rmse.nn3.pca

# hidden = 6
set.seed(6822651)
nn6.pca <- neuralnet(Close ~ PC1+PC2+PC3, data=new.data.train, hidden = 6, err.fct = "sse",  
                     threshold = 0.04, act.fct = "tanh", stepmax = 1e7, linear.output = T)
nn6.pred.pca <- predict(nn6.pca, new.data.test)
rmse.nn6.pca <- rmse(nn6.pred.pca, new.data.test$Close)
rmse.nn6.pca

# hidden = 10
set.seed(6822651)
nn10.pca <- neuralnet(Close ~ PC1+PC2+PC3, data=new.data.train, hidden = 10, err.fct = "sse",  
                      threshold = 0.04, act.fct = "tanh", stepmax = 1e7, linear.output = T)
nn10.pred.pca <- predict(nn10.pca, new.data.test)
rmse.nn10.pca <- rmse(nn10.pred.pca, data.test$Close)
rmse.nn10.pca

# hidden = 14
set.seed(6822651)
nn14.pca <- neuralnet(Close ~ PC1+PC2+PC3, data=new.data.train, hidden = 14, err.fct = "sse",  
                      threshold = 0.04, act.fct = "tanh", stepmax = 1e7, linear.output = T)
nn14.pred.pca <- predict(nn14.pca, new.data.test)
rmse.nn14.pca <- rmse(nn14.pred.pca, new.data.test$Close)
rmse.nn14.pca

# hidden = 15
set.seed(6822651)
nn15.pca <- neuralnet(Close ~ PC1+PC2+PC3, data=new.data.train, hidden = 15, err.fct = "sse",  
                      threshold = 0.04, act.fct = "tanh", stepmax = 1e7, linear.output = T)
nn15.pred.pca <- predict(nn15.pca, new.data.test)
rmse.nn15.pca <- rmse( nn15.pred.pca, new.data.test$Close)
rmse.nn15.pca

# The best Hidden = 2 with rmse=0.03374374

nn.pca <- nn2.pca
nn.pca.pred <- nn2.pred.pca
nn.pca.train <- predict(nn.pca, new.data.train)
rmse.nn.pca <- rmse.nn2.pca
rmse.nn.pca

plot(nn.pca, rep = "best")

#Plot the prediction of the ANN model  wiht pca
plot(data$Close, xlab = "Days", type = "l", ylab="Close", main="Observed, Fitted and predicted Values using PCA-ANN")
lines(x = c(1209:1510), nn.pca.pred, col="green")
lines(x = c(1:1208), nn.pca.train, col="blue")
legend(y = 5.75, x = 5, legend = c("Obs", "pred.pca.test", "pred.pca.train"), col = c("black", "green","blue"), lwd = 1)

#data demoralization
# Results
prediction <- exp(nn.pca.pred)
actual <- exp(data.test$Close)
results3 <- data.frame( actual = actual, Prediction = prediction)
Days <- data.test$Date 
results1 <- cbind(Days, results1)
head(results3)
tail(results3)
