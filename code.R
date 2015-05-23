###########################################################################
## This is the full code file for the Practial Machine Learning course
## Pasi Hyytiäinen, May 2015
## More info https://github.com/LilaLipetti/PMLProject
## 
## Note! doParallel was used to boost up execution time
## Noticed that sometimes the parallel execution didn't work 
## (no errors or warnings noticed).
## RStudio needed to be restarted to get it work, sometimes even 2-3 times.
## 
## Code execution time about 1,5-2h with win7, 16Gb, 6 threads
###########################################################################


######################################################################
##            STEP 1 : read & initialize data
######################################################################

######################################################################
# Data initialization:
# delete workspace
# load library
# set the seed so that analysis can be reproduced if needed
######################################################################
rm(list = ls(all = TRUE))

library(plyr)
library(caret)
library(doParallel)
library(Hmisc)
library(randomForest)
library(C50)
library(gbm)
library(klaR)
library(MASS)



set.seed(1304)


######################################################################
# Load data
# while checking data outside R, noticed that some rows
# contained strange #DIV/0! values => those will be considered
# as 'NA's 
######################################################################

trainDataUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testDataUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainDataFile <- "pml-training.csv"
testDataFile  <- "pml-testing.csv"

if (!file.exists(trainDataFile)) {
        download.file(trainUrl, destfile=trainDataFile, method="curl")
}
if (!file.exists(testDataFile)) {
        download.file(testUrl, destfile=testDataFile, method="curl")
}

trainRawData <- read.csv(trainDataFile,
                         header=TRUE, 
                         as.is = TRUE, 
                         stringsAsFactors = FALSE, 
                         sep=',', 
                         na.strings=c('NA','','#DIV/0!'))

testRawData <- read.csv(testDataFile,
                        header=TRUE, 
                        as.is = TRUE, 
                        stringsAsFactors = FALSE, 
                        sep=',', 
                        na.strings=c('NA','','#DIV/0!'))

dim(trainRawData)
dim(testRawData)


######################################################################
# Clean the data, 
######################################################################
cleanedData<-trainRawData
cleanedData$classe<- as.factor(cleanedData$classe)
cleanedData<-cleanedData[!names(cleanedData) %in% c("X",
                                                       "user_name",
                                                       "raw_timestamp_part_1",
                                                       "raw_timestamp_part_2",
                                                       "cvtd_timestamp",
                                                       "new_window",
                                                       "num_window")]

cleanedData <- cleanedData[, colSums(is.na(cleanedData)) == 0] 


# after cleaning should be 19622 obs, 53 variables
dim(trainRawData)
dim(cleanedData) 


######################################################################
##            STEP 2: create models
######################################################################


######################################################################
# Split the given training data to 2 groups
# training and cross verifiction group
######################################################################
trainingIndex <- createDataPartition(cleanedData$classe, list=FALSE, p=.80)
training<-cleanedData[trainingIndex,]
verifiction <- cleanedData[-trainingIndex,]



######################################################################
# check the near Zero vars, if there is some, those needs to be
# from the datasets
######################################################################
nzv <- nearZeroVar(training,saveMetrics = T)


## as nzv is emtpy nothing to do



######################################################################
## Creating 4 different models
######################################################################

registerDoParallel(6)

######################################################################
# Boosting model 10-fold cross validation to predict classe 
# with all other predictors.
######################################################################
boostControl<-trainControl(method = "cv", number = 10)
modelBoost <- train(classe ~ ., 
                    method = "gbm", 
                    data = training, 
                    verbose = F, 
                    trControl = boostControl)
training_pred_modelBoost <- predict(modelBoost, training) 
cm_modelBoost   <- confusionMatrix(training_pred_modelBoost, training$classe)

plot(modelBoost, ylim = c(0.9, 1))

######################################################################
# method='C5.0'
# Since the training data CSV file seems to be obviously ordered,
# boot-632 needs to be used to get some randomization of the data order
######################################################################
fitControl <- trainControl(method='boot632')
c50Grid <- expand.grid(trials = c(40,50,60,70), 
                       model = c('tree'), 
                       winnow = c(TRUE, FALSE))
modelC50  <- train(classe ~ ., 
                   data = training, 
                   method='C5.0', 
                   trControl=fitControl, 
                   tuneGrid = c50Grid)
training_pred_modelC50 <- predict( modelC50, training) 
cm_modelC50   <- confusionMatrix(training_pred_modelC50, training$classe)

plot(modelC50, ylim = c(0.9, 1))


######################################################################
# Naive Bayes
######################################################################
train_control <- trainControl(method="cv", number=10)

modelNB <- train(classe~.,training,method='nb',
                trControl=train_control)

training_pred_modelNB <- predict(modelNB, training) 
cm_modelNB   <- confusionMatrix(training_pred_modelNB, training$classe)
plot(modelNB)


######################################################################
# Random Foreest : 10-fold cross validation model
######################################################################
train_control <- trainControl(method="cv", number=10)
modelRF <- train(classe~., data=training, 
                   trControl=train_control, method="rf")
training_pred_modelRF <- predict(modelRF, training) 
cm_modelRF2   <- confusionMatrix(training_pred_modelRF, training$classe)

plot(modelRF, ylim = c(0.9, 1))


######################################################################
##            STEP 3: Model accuracy
######################################################################
######################################################################
# Let's check the accuracy of the tested models
######################################################################
trainAccuracy <- data.frame(ModelType=c('C50/boot362','Boost',
                                        ' Naive Bayes ', 'RF 5 folder'), 
                            Accuracy=c(cm_modelC50$overall['Accuracy'],
                                       cm_modelBoost$overall['Accuracy'],
                                       cm_modelNB$overall['Accuracy'],
                                       cm_modelRF2$overall['Accuracy']))



######################################################################
##            STEP 4: Model cross verification
######################################################################
######################################################################
# Out-of-sample accuracy
# Now we test the accuracy of each model on our 
# cross verification  'verifiction' data.
######################################################################

vaccC50<-predict(modelC50,verifiction)
vaccC50cm   <- confusionMatrix(vaccC50, verifiction$classe)

vaccBoost<-predict(modelBoost,verifiction)
vaccBoostcm   <- confusionMatrix(vaccBoost, verifiction$classe)


vaccNBP<-predict(modelNB,verifiction)
vaccNBPcm   <- confusionMatrix(vaccNBP, verifiction$classe)


vaccRFP<-predict(modelRF,verifiction)
vaccRFPcm   <- confusionMatrix(vaccRFP, verifiction$classe)


verificationAccuracy <- data.frame(ModelType=c('C50/boot362', 
                                                'Boost',
                                                'Naive Bayes ', 
                                                'RF 5 folder'), 
                                   Accuracy=c(vaccC50cm$overall['Accuracy']*100,
                                              vaccBoostcm$overall['Accuracy']*100,
                                              vaccNBPcm$overall['Accuracy']*100,
                                              vaccRFPcm$overall['Accuracy']*100),
                                   OutOfSampleError=c(
                                           (1-vaccC50cm$overall['Accuracy'])*100,
                                           (1-vaccBoostcm$overall['Accuracy'])*100,
                                           (1-cm_modelNB$overall['Accuracy'])*100,
                                           (1-vaccRFPcm$overall['Accuracy'])*100
                                           )
                                   )



######################################################################
##            STEP 5 : use the chosen model to prediction
######################################################################

######################################################################
## now we do the prediction for the given test data set
##  write prediction files with a helper function
######################################################################
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_", i, ".txt")
                write.table(x[i], 
                            file = filename, 
                            quote = FALSE, 
                            row.names = FALSE,
                            col.names = FALSE)
        }
}

predictRF<-predict(modelRF,testRawData)
pml_write_files(predictRF)


######################################################################
## Variable importance of the selected method
######################################################################
varImp(modelRF)



######################################################################
##            SOME EXTRAS not really needed 
## Additional prediction with other models, as if we look up the
## verificationAccuracy, only the Naive Bayes might give different
## output and all other models should give same result as the selected
## model
######################################################################
predictC50<-predict(modelC50,testRawData)
predictBoost<-predict(modelBoost,testRawData)
predictNB<-predict(modelNB,testRawData)

