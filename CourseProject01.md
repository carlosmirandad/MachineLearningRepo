

Practical Machine Learning - Course Project
===========================================

## Objectives

We are asked to determine how well individuals perform a weight lifting exercise using human-activity-recognition data captured by accelerometers. 

We are asked to answer the following questions in this report:
- how did we built the predictive model,
- how did we handle cross validation,
- what is the expected out of sample error, and 
- what is the rationale for the choices made

I decided to utilize a random forest to accomplish this.  This model will: (1) predict the level of quality of the exercises, (2) perform the cross-validation, and (3) determine the expected "out of sample error rate" of those predictions. 

## Exploratory Data Analysis

I started with a review of the training data, which should be credited to: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.


```r
training.all <- read.csv("./pml-training.csv", na.strings = c("NA","#DIV/0!"))
testing.all  <- read.csv("./pml-testing.csv",  na.strings = c("NA","#DIV/0!"))
summary(training.all); dim(training.all)
```

The data summary (not printed in this report due to size) immediately revealed multiple variables with a large number of missing values (NA.)  This was a first reason for my selection of random forests. If I were to use a method such as logistic regression instead, I would have to impute values which would be unreliable (since the missing percentages are over 80%), or drop all variables with any missing value, or drop the incomplete cases which would cause information loss.  

Fortunately, methods such as classification trees and random forests have the following advantages:
- they can handle missing values without problem (they can use missing values when splitting a node as naturally as they can use any other categorical or continuous value), 
- they can flexibly handle data following any distribution (e.g. no normality assumptions or similar requirements), 
- they can handle different types of relationships against the predictor (e.g. linear, quadratic, or other) without having to apply transformations, 
- they are easy to interpret and explain, and 
- they have a track record of excelent prediction results.  

For these reasons, I will go with a model from the decision tree or random forest family. I decided to chose a random forest over classification trees due to its superior prediction power (a random forest creates and assemmbles many individual trees.) 

Continuing with the data exploration, I noticed varibles that didn't make sense to include in the model, so I removed them from the datasets:
- Subject identity (which would be relevant if we wanted "personalized" models, but its a bad variable if the model needs to generalize to other subjects)
- Time stamps (which could be used for feature extraction but not for model training since they don't repeat)
- Attributes without variability (amplitude_yaw_belt, amplitude_yaw_dumbbell, amplitude_yaw_forearm)
- Attributes completely missing (kurtosis_yaw_belt, skewness_yaw_belt, kurtosis_yaw_dumbbell, skewness_yaw_dumbbell, kurtosis_yaw_forearm, skewness_yaw_forearm)
- Observation number in the file (irrelevant for prediction)




I also wanted to understand the distribution of the target variable (classe.)  One can see that its faily balanced except for a class A (which represents an exercise well executed):


```r
table(training.all$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```


## Variable Selection

Before starting to develop the real prediction model, I decided to take a small sample and run a "quick" model for variable prioritization purposes. I fed all 147 variables and generated a list of the "top 50 variables" by  importance. The random forest algorythm has a clever mechanism to identify the importance of each variable by substracting "randomly-permuted correct predictions" from "real correct predictions".



```r
set.seed(1)
smallsample    <- createDataPartition(y=training.all[,"classe"], p=0.30, list=FALSE)
training.small <- training.all[smallsample,]
fit.small      <- train(classe~., method="rf", data=training.small)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
var.importance <- varImp(fit.small)$importance
var.importance$names <- row.names(var.importance)
model.vars     <- var.importance[order(var.importance$Overall, decreasing=TRUE),"names"][1:50]
model.vars
```

```
##  [1] "roll_dumbbell"           "var_pitch_belt"         
##  [3] "var_roll_belt"           "avg_roll_dumbbell"      
##  [5] "stddev_roll_belt"        "stddev_pitch_belt"      
##  [7] "magnet_dumbbell_z"       "accel_dumbbell_y"       
##  [9] "avg_pitch_forearm"       "var_total_accel_belt"   
## [11] "var_accel_forearm"       "yaw_dumbbell"           
## [13] "min_roll_forearm"        "amplitude_pitch_belt"   
## [15] "var_accel_dumbbell"      "num_window"             
## [17] "pitch_belt"              "magnet_belt_y"          
## [19] "accel_forearm_x"         "magnet_belt_z"          
## [21] "avg_pitch_belt"          "max_picth_forearm"      
## [23] "var_accel_arm"           "amplitude_roll_belt"    
## [25] "accel_arm_y"             "gyros_forearm_x"        
## [27] "avg_pitch_dumbbell"      "max_roll_forearm"       
## [29] "skewness_roll_forearm"   "pitch_forearm"          
## [31] "magnet_belt_x"           "var_pitch_forearm"      
## [33] "var_yaw_arm"             "amplitude_pitch_forearm"
## [35] "accel_dumbbell_x"        "stddev_pitch_forearm"   
## [37] "max_picth_arm"           "stddev_yaw_belt"        
## [39] "amplitude_yaw_arm"       "max_yaw_arm"            
## [41] "avg_roll_forearm"        "max_roll_dumbbell"      
## [43] "roll_forearm"            "min_roll_belt"          
## [45] "amplitude_roll_forearm"  "magnet_forearm_z"       
## [47] "var_pitch_arm"           "amplitude_roll_arm"     
## [49] "stddev_yaw_arm"          "amplitude_pitch_arm"
```


## Model Building

After identifying the most important variables for the prediction of the class, I created the "real" random forest using the top variables and all the available observations. This is the predictive model that we'll use to score the observations whose class we don't know.

Here is the rationale for my choices: 
- I decided to use every observation as opposed to creating two partitions (e.g. training vs. testing) because the random forest handles this internally (as part of the algorythm itself) when it creates trees.
- Having less variables (but the most important ones) will help the random forest to finish faster without sacrificing much prediction power, so I kept only the target variable and the "important variables" listed above in the input dataset. 
- I let the random forest handle the cross-validation, since it has the ability to do it internally and does it very well. The forest will even give us the "OOB estimate of  error rate" (OOB=Out of Bag.)  This OOB error rate is, in practice, a fairly reliable estimate of the error rate that we will observe when we apply the prediction model to new data (e.g. the out-of-sample error rate that we need to determine.)


```r
set.seed(1)
fit <- train(classe~., method="rf", data=training.all[,c("classe",model.vars)])
fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 13.31%
## Confusion matrix:
##    A  B  C  D  E class.error
## A 80  0  2  2  1     0.05882
## B  9 50  3  2  0     0.21875
## C  2  3 49  1  1     0.12500
## D  2  1  3 55  0     0.09836
## E  1  3  3  4 46     0.19298
```





## Conclusions

- The regression forest performs well predicting the target variable classe. It could certainly be improved (like any model) but for the purpose of this submission it is more than adequate.  
- The model (the "fit" object) can be used to predict "classe"" for the observations in the testing dataset (which I will submit via the appropiate screen) 
- We estimate that the out of sample error rate will be around 13% (the value from the random forest results table above.)

With this, we have completed all the objectives of the report.  


Report date:

```
## [1] "2014-06-22 08:37:33 CDT"
```
