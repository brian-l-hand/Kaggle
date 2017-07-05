###Used to generate a submission for Two Sigma Connect: Rental Listing Inquiries on Kaggle.com
###This program takes the output data from Generate_Features_TwoSigma.R and uses the data as input in a gradient boosting model


rm(list=ls())
#setwd("\\users\\brian\\OneDrive\\Documents\\Kaggle\\TwoSigma")
setwd("\\users\\brian\\OneDrive\\Documents\\Grad_School\\Spring_2017\\297\\Extra_Assignment")
require(Matrix)
require(xgboost)
load("trainDataFeatures.RData")
options(na.action='na.pass')

X <- subset(trainDataFeatures, select = -c(interest_level, cluster_assign))
X[] <- lapply(X, as.numeric)
cluster_binaries <- sparse.model.matrix(interest_level ~ 0 + cluster_assign, data = trainDataFeatures)
X <- cbind(X, as.matrix(cluster_binaries))
y <- as.numeric(trainDataFeatures$interest_level) - 1

trainIndex = sample(length(y), floor(0.8*length(y)))
XTrain <- X[trainIndex,] 
XValidate <- X[-trainIndex,] 
yTrain <- y[trainIndex]
yValidate <- y[-trainIndex]

load("testDataFeatures.RData")
XTest <- subset(testDataFeatures, select = -c(cluster_assign))
XTest[] <- lapply(XTest, as.numeric)
cluster_binaries <- sparse.model.matrix(~ 0 + cluster_assign, data = testDataFeatures) 
XTest <- cbind(XTest, as.matrix(cluster_binaries))



#convert to DMatrix format that xgboost uses
trainDMatrix <- xgb.DMatrix(as.matrix(XTrain), label = yTrain)
validateDMatrix <- xgb.DMatrix(as.matrix(XValidate), label = yValidate)
testDMatrix <- xgb.DMatrix(as.matrix(XTest))




#set.seed(123)
xgb.params = list(
  colsample_bytree= 0.7,
  subsample = 0.7,
  eta = 0.01, #0.01 #0.1
  objective= 'multi:softprob',
  max_depth = 6, #8 #6 #4
  min_child_weight= 1,
  eval_metric= "mlogloss",
  num_class = 3
)
#treedepth 5: 0.547437
#treedepth 6: 0.534226
#treedepth 7: :0.545065

#eta = 0.01 0.537814
#perform training
model.xgb <- xgb.train(params = xgb.params, data = trainDMatrix, nrounds = 2000, 
                      watchlist = list(train = trainDMatrix, val = validateDMatrix),
                      print_every_n = 25, early_stopping_rounds=50)



pred <- matrix(data = predict(model.xgb, newdata = testDMatrix), nrow = nrow(testDataFeatures), ncol = 3, byrow = TRUE)
submissionData <- cbind(pred, testDataFeatures$listing_id)
colnames(submissionData) <- c("low", "medium", "high", "listing_id")
# change the order
submissionData=submissionData[,c(4,3,2,1)]
#write.csv(submission.data, paste(Sys.time(),".csv", sep=""),row.names = FALSE)
write.csv(submissionData, "Kaggle_Submission.csv",row.names = FALSE)


imp <- xgb.importance(colnames(trainDMatrix),model = model.xgb)
xgb.ggplot.importance(imp[1:20])
xgb.ggplot.importance(imp[21:40])