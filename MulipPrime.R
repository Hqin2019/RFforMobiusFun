rm(list=ls(all=TRUE)) #Clear the environment.
#set up the working file.
#setwd("C:/MultiPrimeMu")
#install.packages("primes")
#install.packages("numbers")
#install.packages("neuralnet")
#install.packages("nnet")

options(max.print = .Machine$integer.max)

#No need to run this part repeatedly.
#For prime numbers up to 265, get all numbers with single primes, product of two primes
#and product of three primes.
library(primes)
p<- generate_primes(max=265)
p2<- unlist(lapply(combn(p, 2, simplify = FALSE), prod))
p3<- unlist(lapply(combn(p, 3, simplify = FALSE), prod))
p4<- unlist(lapply(combn(p, 4, simplify = FALSE), prod))
p_val<- c(p, p2, p3, p4) 
length(p_val)
#[1] 396606

#compute p_val mod 4, 9, 25, and 49
p_mod4<- p_val %% 4
p_mod9<- p_val %% 9
p_mod25<- p_val %% 25
p_mod49<- p_val %% 49

#Compute corresponding mu for p_val
library(numbers)
mu<- sapply(p_val, moebius)
factor_mu<- as.factor(mu)
#relabel the levels of factor_mu
#levels(factor_mu)[levels(factor_mu)=="-1"]<- 'o'
#levels(factor_mu)[levels(factor_mu)=="1"]<- 'e'
table(factor_mu)
#factor_mu
#-1      1 
#27776 368830 

#create the corresponding data frame
data<- data.frame(n=p_val, mu=factor_mu, mod4=p_mod4, mod9=p_mod9, mod25=p_mod25, mod49=p_mod49)
#Since the length is too long, computing mu values takes a lot of time. Save the data for future ues.
save(data, file="primeM.Rda")
save(mu, file="mu_value.Rda")
#End of no-repeating part

#Star from this line by loading the created dataframe.
load("primeM.Rda")
str(data)
table(data$mu)
#-1      1 
#27776 368830 
# "-1" == "o", "1"=="e"

#Divide the data into training data and test data with 20-80 principle.
set.seed(1)
index<- sample(nrow(data), floor(nrow(data)*0.8))
#317284 obs for training data and 79322 obs for test data
train<- data[index, ]
table(train$mu)
#-1      1 
#22293 294991 
test<- data[-index, ]
table(test$mu)
#-1     1 
#5483 73839

#Neural Network by "neuralnet" library
library(neuralnet)
library(nnet)
set.seed(2)
n<- names(train)
f<- as.formula(paste("mu~", paste(n[!n %in% "mu"], collapse = "+")))
nn<- neuralnet(f, data=train, hidden=c(3, 5), act.fct = "logistic", algorithm = "backprop",learningrate =0.02,linear.output = FALSE, lifesign = "minimal")
      
plot(nn)


#Check the accuracy on the training set
pr.nn<- compute(nn, train[, -2])
pr.nn_<- pr.nn$net.result
head(pr.nn_)
original_values<- max.col(train[, 2])
pr.nn_2<- max.col(pr.nn_)
mean(pr.nn_2 == train[, 2])


#Compute accuracy on the test data set. max.col is used to find the maximum position for each row of a matrix
pr.nn_test <- compute(nn, test[, -2])
# Extract results 
pr.nn_test_ <- pr.nn_test$net.result
# Accuracy (test set)
original_values_test <- max.col(test[, 2])
pr.nn_2_test <- max.col(pr.nn_test_)
#paccuracy is the prediction accuracy.
paccuracy<- mean(pr.nn_2_test == test[, 2])
paccuracy
MSE.nn<- sum((as.numeric(test[, 2]) - pr.nn_test_)^2)/nrow(test)

#try caret package, with smote sampling.
library(caret)
set.seed(42)
levels(train$mu)<- c("P", "N")
levels(test$mu)<- c("P", "N")
ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 3, 
                     verboseIter = FALSE,
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     sampling = "smote")
model_rf_under <- caret::train(mu ~ .,
                               data = train,
                               method = "rf",
                               preProcess = c("scale", "center"),
                               trControl = ctrl)
final <- predict(model_rf_under, newdata = test, type = "prob")
final$predict <- ifelse(final[, 1] > 0.5, "P", "N")
cm_original <- confusionMatrix(as.factor(final$predict), test$mu)
cm_F<- confusionMatrix(as.factor(final$predict), test$mu, mode = "prec_recall")
#Confusion Matrix and Statistics

#Reference
#Prediction    -1     1
#        -1  4876  5555
#         1    680 68211

#Accuracy : 0.9214          
#95% CI : (0.9195, 0.9233)
#No Information Rate : 0.93            
#P-Value [Acc > NIR] : 1               

#Kappa : 0.5708          

#Mcnemar's Test P-Value : <2e-16          
                                          
#            Sensitivity : 0.87761         
#            Specificity : 0.92469         
#         Pos Pred Value : 0.46745         
#         Neg Pred Value : 0.99013         
#             Prevalence : 0.07004         
#         Detection Rate : 0.06147         
#   Detection Prevalence : 0.13150         
#      Balanced Accuracy : 0.90115         
                                          
#       'Positive' Class : -1              
                                     
#Tuning
#2/14/2023
control <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 3,
                     sampling = "smote",
                     search = "grid")
tunegrid<- expand.grid(.mtry = (1:5))
model_rf_tune <- train(mu ~ .,
                       data = train,
                       method = "rf",
                       preProcess = c("scale", "center"),
                       trControl = control,
                       tuneGrid = tunegrid)
#Prediction with the tuned model
final_tuned <- predict(model_rf_tune, newdata = test, type = "prob")
final_tuned$predict <- ifelse(final_tuned[, 1] > 0.5, "-1", "1")
cm_original_tuned <- confusionMatrix(as.factor(final_tuned$predict), test$mu)
cm_F_tuned<- confusionMatrix(as.factor(final_tuned$predict), test$mu, mode = "prec_recall")

model_rf_tune$times




library(doParallel)
cores <- makeCluster(detectCores()-2)
registerDoParallel(cores = cores)

#Manual search by create 10 folds and repeat 5 times
control <- trainControl(method = 'repeatedcv',
                        number = 10,
                        repeats = 5,
                        search = 'grid',
                        sampling = "rose")
#create tunegrid
tunegrid <- expand.grid(.mtry = c(sqrt(ncol(train))))
modellist <- list()

#train with different ntree parameters
for (ntree in c(15,50,100,250,500)){
  fit <- train(mu ~ .,
               data = train,
               method = 'rf',
               preProcess = c("scale", "center"),
               metric = 'Accuracy',
               tuneGrid = tunegrid,
               trControl = control,
               ntree = ntree)
   key <- toString(ntree)
  modellist[[key]] <- fit
}
 print(fit)

#Compare results
results <- resamples(modellist)
summary(results)

#Random Forest 

#317284 samples
#5 predictor
#2 classes: '-1', '1' 

#Pre-processing: scaled (5), centered (5) 
#Resampling: Cross-Validated (10 fold, repeated 5 times) 
#Summary of sample sizes: 285555, 285555, 285555, 285556, 285555, 285556, ... 
#Addtional sampling using SMOTE prior to pre-processing

#Resampling results:
  
#  Accuracy  Kappa   
#0.904947  0.527984

#Tuning parameter 'mtry' was held constant at a value
#of 2.44949


#Call:
#  summary.resamples(object = results)

#Models: 15, 50, 100, 250, 500 
#Number of resamples: 50 

#Accuracy 
#         Min.   1st Qu.    Median      Mean   3rd Qu. 
#15  0.8977874 0.9021842 0.9033819 0.9037663 0.9058403
#50  0.9000284 0.9028484 0.9040453 0.9040645 0.9051003
#100 0.9012544 0.9031305 0.9048790 0.9046885 0.9064233
#250 0.8968104 0.9038625 0.9055566 0.9050838 0.9064027
#500 0.9010999 0.9038025 0.9050853 0.9049470 0.9063389
#         Max. NA's
#15  0.9077786    0
#50  0.9093545    0
#100 0.9080308    0
#250 0.9086926    0
#500 0.9099218    0

#Kappa 
#         Min.   1st Qu.    Median      Mean   3rd Qu.
#15  0.5057976 0.5169635 0.5205455 0.5219570 0.5271131
#50  0.5124662 0.5214973 0.5246521 0.5250443 0.5283310
#100 0.5154488 0.5221851 0.5269800 0.5270575 0.5320649
#250 0.5043859 0.5246420 0.5297514 0.5285821 0.5331977
#500 0.5158885 0.5240210 0.5279553 0.5279840 0.5318566
#        Max. NA's
#15  0.5356660    0
#50  0.5417267    0
#100 0.5399512    0
#250 0.5412094    0
#500 0.5435943    0


#Keras+Tensorflow with class_weights.
library(dplyr)
library(keras) 
library(tfruns) 
library(tfestimators)


#data preprocessing.
train_x<- train[, -2]
train_y<- train$mu #factors 1 and -1
table(train_y)
train_y #imbalanced data
#o      e 
#22220 295064 

#colnames(mobius_x) <- paste0("V", 1:ncol(mobius_x))
#Standardize features in three way, need comparisons. 
scale_train_x<- scale(train_x)
norm_train_x<- normalize(as.matrix(train_x))

normalize_mm <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}
maxmin_train_x <- as.data.frame(lapply(train_x, normalize_mm))

# One-hot encode response. -1/odd is labeled 0; 1/even is labeled 1.
levels(train_y)[levels(train_y) == "o"]<- 0
levels(train_y)[levels(train_y) == "e"]<- 1
onehot_train_y <- to_categorical(train_y, 2)


test_x<- test[, -2]
test_y<- test[, 2]
#colnames(tmobius_x) <- paste0("V", 1:ncol(tmobius_x))
#Standardize features.
scale_test_x<- scale(test_x)
norm_test_x<- normalize(as.matrix(test_x))
maxmin_test_x <- as.data.frame(lapply(test_x, normalize_mm))

levels(test_y)[levels(test_y) == "o"]<- 0
levels(test_y)[levels(test_y) == "e"]<- 1

onehot_test_y<- to_categorical(test_y, 2)
#onehot_test_y <- to_categorical(test_y, 2)

model <- keras_model_sequential() %>%
  
  # Network architecture
  layer_dense(units = 128, activation = "relu", input_shape = ncol(scale_train_x)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 2, activation = "softmax") %>%
  
  # Backpropagation
  compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_adam(),
    metrics = 'accuracy'
  )

fit1 <- model %>%
  fit(
    x = scale_train_x,
    y = onehot_train_y,
    epochs = 100,
    batch_size = 32,
    validation_split =0.2,
    class_weight = list("0" = 1, "1" = 13.3),
    verbose = FALSE
  )
 fit1
plot(fit1)

# Compute probabilities and predictions on test set
predictions <-  predict_classes(model, scale_test_x)
probabilities <- predict_proba(model, stan_tmobius_x)
table(predictions)

# Evaluate on test data and labels
score <- model %>% evaluate(scale_test_x, onehot_test_y, batch_size = 32)
