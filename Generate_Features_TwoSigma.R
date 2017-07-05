#This program generates features for use as input in the GB_TwoSigma.R program

###Ver3 - only use relativeprice2
##############PART 0 ##############
require(rjson)
require(reshape2)
require(syuzhet)
require(DT)
require(nnet)
require(doParallel)
require(lubridate)
require(FNN)
require(foreach)
rm(list=ls())
registerDoParallel(cores=2)
setwd("\\users\\brian\\OneDrive\\Documents\\Kaggle\\TwoSigma")
trainData <- fromJSON(file='train.json')
testData <- fromJSON(file='test.json')
summary(trainData)

###################################



######Parameters#######
sample_threshold = 20
######################

############PART 1 - Generate Train Features#################

if (file.exists("newTrain.RData")){
  load("newTrain.RData")
} else {
  newTrain <- data.frame(matrix(ncol = 0, nrow = length(trainData$listing_id)))
  attach(trainData)
  newTrain$listing_id <- melt(listing_id)$value
  newTrain$interest_level <- factor(melt(interest_level)$value, levels = c("low", "medium", "high"), ordered = FALSE)
  newTrain$bathrooms <- melt(bathrooms)$value
  newTrain$bedrooms <- melt(bedrooms)$value
  newTrain$price <- as.numeric(melt(price)$value)
  newTrain$longitude <- melt(longitude)$value
  newTrain$latitude <- melt(latitude)$value
  newTrain$manager_id <- melt(manager_id)$value
  newTrain$building_id <- melt(building_id)$value
  newTrain$hour <- factor(hour(melt(created)$value), ordered = FALSE)
  newTrain$weekday <- factor(wday(melt(created)$value, label = TRUE), ordered = FALSE)
  newTrain$week <- factor(week(melt(created)$value), ordered = FALSE)
  newTrain$month <- factor(month(melt(created)$value, label = TRUE), ordered = FALSE)
  detach(trainData)
  save(newTrain, file = "newTrain.RData")
}
attach(newTrain)
###Some listings seem to have a purchase price insted of a monthly rental rate. 
#Assign those observations the expected price
linearFit4 <- lm(price ~ bedrooms + I(bedrooms^2) + I(bedrooms^3) + I(bedrooms^4) + bathrooms + I(bathrooms^2) +I(bathrooms^3) + I(bathrooms^4), data = newTrain)
newTrain$predictedPrice <- predict(linearFit4, newdata = newTrain)
newTrain$price[which(newTrain$predictedPrice*100< price)] <- NA
newTrain <- subset(newTrain, select = -c(predictedPrice))


newTrain$pricePerBedroom <- price/bedrooms
newTrain$pricePerBedroom[newTrain$bedrooms == 0] <- price[newTrain$bedrooms == 0]

newTrain$pricePerBathroom <- price/bathrooms
newTrain$pricePerBathroom[newTrain$bathrooms == 0] <- price[newTrain$bathrooms == 0]

newTrain$pricePerRoom <- price/(bedrooms + bathrooms)
newTrain$pricePerRoom[(newTrain$bedrooms == 0) & (newTrain$bathrooms == 0)] <- price[(newTrain$bedrooms == 0) & (newTrain$bathrooms == 0)]


#Building_ID Dummies
newTrain$building_id_zero <- as.integer(building_id == 0)

#Manager Skill:
table1 <- aggregate(interest_level ~ manager_id, newTrain, function(x) table(x)/length(x))
table2 <- aggregate(interest_level ~ manager_id, newTrain, function(x) length(x))
table <- cbind(table1, table2[,2])
colnames(table)[3] <- "Total"

#manager Skill = high% times 2 plus medium% 
table$manager_skill <- (table[,2][,3])*2 + table[,2][,2]
#avg_skill <- mean(table$manager_skill[which(table$Total>=sample_threshold])

for(i in c(1:nrow(table))){
  if (table$Total[i] < sample_threshold) {
    table$manager_skill[i] <- NA
  }
}
save(table, file = "manager_skill_table.RData")
newTrain <- merge(newTrain, table[,c(1,4)], by = 'manager_id')
newTrain <- newTrain[order(newTrain$listing_id),]


###Building Interest:
table1 <- aggregate(interest_level ~ building_id, newTrain, function(x) table(x)/length(x))
table2 <- aggregate(interest_level ~ building_id, newTrain, function(x) length(x))
table <- cbind(table1, table2[,2])
colnames(table)[3] <- "Total"

#Building_interest = high% times 2 plus medium% 
table$building_interest <- (table[,2][,3])*2 + table[,2][,2]
#avg_interest <- mean(table$building_interest[which(table$Total>=sample_threshold)])

for(i in c(1:nrow(table))){
  if (table$Total[i] < sample_threshold) {
    table$building_interest[i] <- NA
  }
  if (table$building_id[i] == '0'){
    table$building_interest[i] <- NA
  }
}
save(table, file = "building_interest_table.RData")

newTrain <- merge(newTrain, table[,c(1,4)], by = 'building_id')
newTrain <- newTrain[order(newTrain$listing_id),]

detach(newTrain)
###Features Analysis
if (file.exists("topFeatures.RData")){
  load("topFeatures.RData")
} else {
  agg_words <- c(unlist(trainData$features[1]))
  for (i in c(2:length(trainData$description))){
    agg_words <- c(agg_words, unlist(trainData$features[i]))
  }
  sort(table(agg_words), decreasing=T)[1:50]
  names(sort(table(agg_words), decreasing=T)[1:50])
  topFeatures <- c(names(sort(table(agg_words), decreasing=T)[1:50]))
  save(topFeatures, file = "topFeatures.RData")
}
featNames <- c(rep(NA, length(topFeatures)))
for (i in c(1:length(topFeatures))){
  featNames[i] <-  gsub('([[:punct:]])|\\s+','_',topFeatures[i])
}
featNames <- c("listing_id", featNames)


if (file.exists("trainFeatures.RData")){
  load("trainFeatures.RData")
} else {
  features <- data.frame(matrix(0, ncol = length(topFeatures)+1 , nrow = length(trainData$description)))
  colnames(features) <- featNames
  for (i in c(1:length(trainData$features))){
    features[i,1] <- newTrain$listing_id[i]
    for (j in c(1:length(topFeatures))){
      word <- topFeatures[j]
      if(word %in% unlist(trainData$features[i])){
        features[i,j+1] <- 1
      }
    }
  }
  save(features, file = "trainFeatures.RData")
}

attach(features)
###Combine redundant features
#Laundry_in_Building, Laundry_In_Building
features$Laundry_in_Building <- Laundry_in_Building + Laundry_In_Building
#Hardwood_Floors, HARDWOOD
features$Hardwood_Floors <- Hardwood_Floors + HARDWOOD
#Pre_War, prewar, Prewar,
features$Pre_War <- Pre_War + prewar + Prewar
#Dishwasher, dishwasher
features$Dishwasher <- Dishwasher + dishwasher
#Elevator, elevator
features$Elevator <- Elevator + elevator
#Roof_Deck, Roof_deck
features$Roof_Deck <- Roof_Deck + Roof_deck
#Swimming_Pool, Pool
features$Swimming_Pool <- Swimming_Pool + Pool
#High_Ceilings, HIGH_CEILINGS
features$High_Ceilings <- High_Ceilings + HIGH_CEILINGS

#Check that combined features are less than or equal to 1  
max(Laundry_in_Building, Hardwood_Floors, Pre_War, Dishwasher, Elevator, Roof_Deck, Swimming_Pool, High_Ceilings)<=1
#All clear!

#Delete redundant variables
features <- subset(features, select = -c(Laundry_In_Building, HARDWOOD, prewar, Prewar, dishwasher, elevator, Roof_deck, Pool, HIGH_CEILINGS))

####Questionable combinations
#Laundry_In_Building, Laundry_Room ???
#Common_Outdoor_Space, PublicOutdoor ???

newTrain <- merge(newTrain, features, by ='listing_id') 
detach(features)


###description Word count
for (i in c(1:length(trainData$description))){
  newTrain$desc_word_count[i] <- sapply(gregexpr("\\W+", trainData$description[i]), length) + 1
}


###feature count
for (i in c(1:length(trainData$photos))){
  newTrain$feature_count[i] <- length(unlist(trainData$features[i]))
}

###Photo Count
for (i in c(1:length(trainData$photos))){
  newTrain$photo_count[i] <- length(unlist(trainData$photos[i]))
}

###sentiment analysis
if (file.exists("trainSentiment.RData")){
  load("trainSentiment.RData")
} else {
  sentiment<- data.frame(matrix(ncol = 11, nrow = length(trainData$description)))
  colnames(sentiment) <- c("listing_id", "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "negative", "positive")
  for (i in c(1:length(trainData$description))){
    sentiment[i,] <- cbind(newTrain$listing_id[i], get_nrc_sentiment(as.character(trainData$description[i])))
  }
  save(sentiment, file = "trainSentiment.RData")
}

newTrain <- merge(newTrain, sentiment, by ='listing_id')

#Drop variables we don't want in regression
trainDataFeatures <- subset(newTrain, select = -c(manager_id, building_id))
save(trainDataFeatures, file = "trainDataFeaturesIncomplete.RData")
##################################################################




################PART 2 - Generate Test Features#############
#######Build Test features
if (file.exists("newTest.RData")){
  load("newTest.RData")
} else {
  newTest <- data.frame(matrix(ncol = 0, nrow = length(testData$listing_id)))
  attach(testData)
  newTest$listing_id <-melt(listing_id)$value
  newTest$bathrooms <- melt(bathrooms)$value
  newTest$bedrooms <- melt(bedrooms)$value
  newTest$price <- as.numeric(melt(price)$value)
  newTest$longitude <- melt(longitude)$value
  newTest$latitude <- melt(latitude)$value
  newTest$manager_id <- melt(manager_id)$value
  newTest$building_id <- melt(building_id)$value
  newTest$hour <- factor(hour(melt(created)$value), ordered = FALSE)
  newTest$weekday <- factor(wday(melt(created)$value, label = TRUE), ordered = FALSE)
  newTest$week <- factor(week(melt(created)$value), ordered = FALSE)
  newTest$month <- factor(month(melt(created)$value, label = TRUE), ordered = FALSE)
  detach(testData)
  save(newTest, file = "newTest.RData")
}
attach(newTest)

###Some listings seem to have a purchase price insted of a monthly rental rate. 
#Assign those observations the expected price
linearFit4 <- lm(price ~ bedrooms + I(bedrooms^2) + I(bedrooms^3) + I(bedrooms^4) + bathrooms + I(bathrooms^2) +I(bathrooms^3) + I(bathrooms^4), data = newTest)
newTest$predictedPrice <- predict(linearFit4, newdata = newTest)
newTest$price[which(newTest$predictedPrice*100< price)] <-newTest$predictedPrice[which(newTest$predictedPrice*100< price)]
newTest <- subset(newTest, select = -c(predictedPrice))

newTest$pricePerBedroom <- price/bedrooms
newTest$pricePerBedroom[newTest$bedrooms == 0] <- price[newTest$bedrooms == 0]

newTest$pricePerBathroom <- price/bathrooms
newTest$pricePerBathroom[newTest$bathrooms == 0] <- price[newTest$bathrooms == 0]

newTest$pricePerRoom <- price/(bedrooms + bathrooms)
newTest$pricePerRoom[(newTest$bedrooms == 0) & (newTest$bathrooms == 0)] <- price[(newTest$bedrooms == 0) & (newTest$bathrooms == 0)]


newTest$building_id_zero <- as.integer(building_id == 0)

####manger_skill
load("manager_skill_table.RData")
#avg_skill <- mean(table$manager_skill[which(table$Total>=sample_threshold)])
newTest <- merge(newTest, table[,c(1,4)], by = 'manager_id', all.x = TRUE)
if(anyNA(newTest$manager_skill)){
  for (i in c(which(is.na(newTest$manager_skill)))){
    newTest$manager_skill[i] <- NA
  }
}
newTest <- newTest[order(newTest$listing_id),]

####building_interest
load("building_interest_table.RData")
#avg_interest <- mean(table$building_interest[which(table$Total>=sample_threshold)])
newTest <- merge(newTest, table[,c(1,4)], by = 'building_id', all.x = TRUE)
if(anyNA(newTest$building_interest)){
  for (i in c(which(is.na(newTest$building_interest)))){
    newTest$building_interest[i] <- NA
  }
}
newTest <- newTest[order(newTest$listing_id),]
detach(newTest)

###Features Analysis
load("topFeatures.RData")
featNames <- c(rep(NA, length(topFeatures)))
for (i in c(1:length(topFeatures))){
  featNames[i] <-  gsub('([[:punct:]])|\\s+','_',topFeatures[i])
}
featNames <- c("listing_id", featNames)

if (file.exists("testFeatures.RData")){
  load("testFeatures.RData")
} else {
  features <- data.frame(matrix(0, ncol = length(topFeatures)+1 , nrow = length(testData$features)))
  colnames(features) <- featNames
  for (i in c(1:length(testData$features))){
    features[i,1] <- newTest$listing_id[i]
    for (j in c(1:length(topFeatures))){
      word <- topFeatures[j]
      if(word %in% unlist(testData$features[i])){
        features[i,j+1] <- 1
      }
    }
  }
  save(features, file = "testFeatures.RData")
}

attach(features)
###Combine redundant features
#Laundry_in_Building, Laundry_In_Building
features$Laundry_in_Building <- Laundry_in_Building + Laundry_In_Building
#Hardwood_Floors, HARDWOOD
features$Hardwood_Floors <- Hardwood_Floors + HARDWOOD
#Pre_War, prewar, Prewar,
features$Pre_War <- Pre_War + prewar + Prewar
#Dishwasher, dishwasher
features$Dishwasher <- Dishwasher + dishwasher
#Elevator, elevator
features$Elevator <- Elevator + elevator
#Roof_Deck, Roof_deck
features$Roof_Deck <- Roof_Deck + Roof_deck
#Swimming_Pool, Pool
features$Swimming_Pool <- Swimming_Pool + Pool
#High_Ceilings, HIGH_CEILINGS
features$High_Ceilings <- High_Ceilings + HIGH_CEILINGS

#Check that combined features are less than or equal to 1  
max(Laundry_in_Building, Hardwood_Floors, Pre_War, Dishwasher, Elevator, Roof_Deck, Swimming_Pool, High_Ceilings)<=1
#All clear

#Delete redundant variables
features <- subset(features, select = -c(Laundry_In_Building, HARDWOOD, prewar, Prewar, dishwasher, elevator, Roof_deck, Pool, HIGH_CEILINGS))

####Questionable combinations
#Laundry_In_Building, Laundry_Room ???
#Common_Outdoor_Space, PublicOutdoor ???

newTest <- merge(newTest, features, by ='listing_id') 
detach(features)

attach(newTest)
###description Word count
for (i in c(1:length(testData$description))){
  newTest$desc_word_count[i] <- sapply(gregexpr("\\W+", testData$description[i]), length) + 1
}


###feature count
for (i in c(1:length(testData$features))){
  newTest$feature_count[i] <- length(unlist(testData$features[i]))
}

###Photo Count
for (i in c(1:length(testData$photos))){
  newTest$photo_count[i] <- length(unlist(testData$photos[i]))
}

###sentiment analysis
if (file.exists("testSentiment.RData")){
  load("testSentiment.RData")
} else {
  sentiment<- data.frame(matrix(ncol = 11, nrow = length(testData$description)))
  colnames(sentiment) <- c("listing_id", "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "negative", "positive")
  for (i in c(1:length(testData$description))){
    sentiment[i,] <- cbind(newTest$listing_id[i], get_nrc_sentiment(as.character(testData$description[i])))
  }
  save(sentiment, file = "testSentiment.RData")
}
newTest <- merge(newTest, sentiment, by ='listing_id')
testDataFeatures <- subset(newTest, select = -c(manager_id, building_id))
save(testDataFeatures, file = "testDataFeaturesIncomplete.RData")
detach(newTest)
#########################################################


###############Spatial Analysis##########################

###Parameters#####
numNeighbors = 20
numClusters = 60
#################

load("trainDataFeaturesIncomplete.RData")
load("testDataFeaturesIncomplete.RData")

combined <- data.frame(matrix(ncol = 0, nrow = (length(testDataFeatures$listing_id) + length(trainDataFeatures$listing_id))))
combined$listing_id <- c(testDataFeatures$listing_id, trainDataFeatures$listing_id)
combined$longitude <- c(testDataFeatures$longitude, trainDataFeatures$longitude)
combined$latitude  <- c(testDataFeatures$latitude, trainDataFeatures$latitude)
combined$price  <- c(testDataFeatures$price, trainDataFeatures$price)
combined$bedrooms <- c(testDataFeatures$bedrooms, trainDataFeatures$bedrooms)
combined$bathrooms <- c(testDataFeatures$bathrooms, trainDataFeatures$bathrooms)
attach(combined)


linearFit4 <- lm(price ~ bedrooms + I(bedrooms^2) + I(bedrooms^3) + I(bedrooms^4) + bathrooms + I(bathrooms^2) +I(bathrooms^3) + I(bathrooms^4), data = combined, na.action = na.exclude)


notna <- which(!is.na(price))
#combined$relativePrice1 <- NA
#combined$relativePrice1[notna] <- linearFit4$residuals


# rSquaredKNN <- foreach(i=1:50,  .combine='c',  .packages="FNN") %dopar% {
#   knn.reg(train = cbind(combined$longitude, combined$latitude),  y = combined$price, k = i)$R2Pred
# }
# which.max(rSquaredKNN)
#k=7 is optimal

combined$relativePrice2 <- knn.reg(train = cbind(combined$longitude, combined$latitude),  y = combined$price, k = numNeighbors)$residuals

# rSquaredKNN <- foreach(i=1:50,  .combine='c',  .packages="FNN") %dopar%{
#   knn.reg(train = cbind(combined$longitude, combined$latitude),  y = combined$relativePrice1, k = i)$R2Pred
# }
# which.max(rSquaredKNN)
#k=9 is optimal

#combined$relativePrice3 <- knn.reg(train = cbind(combined$longitude, combined$latitude),  y = combined$relativePrice1, k = numNeighbors)$residuals


# withinSSKMeans <- foreach(i=1:100,  .combine='c') %dopar% {
#   kmeans(cbind(combined$longitude,combined$latitude), centers = i, iter.max=25)$tot.withinss
# }
# k=30 seem like a good number

clusterFit <- kmeans(cbind(combined$longitude,combined$latitude), centers = numClusters, iter.max=30)
combined$cluster_assign <- factor(clusterFit$cluster)


###Add "Magic" feature

image_time = read.csv("listing_image_time.csv") 
colnames(image_time) <- c("listing_id", "image_time_stamp")

trainDataFeatures<-merge(trainDataFeatures, combined[,c(1,7:8)], by = "listing_id")
testDataFeatures<-merge(testDataFeatures, combined[,c(1,7:8)], by = "listing_id")

trainDataFeatures<-merge(trainDataFeatures, image_time, by = "listing_id")
testDataFeatures<-merge(testDataFeatures, image_time, by = "listing_id")

save(trainDataFeatures,file="trainDataFeatures.RData")
save(testDataFeatures,file="testDataFeatures.RData")
detach(combined)
