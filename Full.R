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
table(factor_mu)
#factor_mu
#-1      1 
#27776 368830 

#create the corresponding data frame
data<- data.frame(n=p_val, mu=factor_mu, mod4=p_mod4, mod9=p_mod9, mod25=p_mod25, mod49=p_mod49)
#Since the length is too long, computing mu values takes a lot of time. Save the data for future ues.
#save(data, file="primeM.Rda")
#save(mu, file="mu_value.Rda")
#Export csv. file
#write.csv(data, file="data.csv")
#End of no-repeating part

#Start from this line by loading the created dataframe.
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

#Random Forests by caret package, with "repeatedcv" method and smote sampling.
library(caret)
set.seed(42)
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
#Making prediction
final <- predict(model_rf_under, newdata = test, type = "prob")
final$predict <- ifelse(final[, 1] > 0.5, "P", "N")
#Create confusion matrices
cm_original <- confusionMatrix(as.factor(final$predict), test$mu)
cm_F<- confusionMatrix(as.factor(final$predict), test$mu, mode = "prec_recall")
    
                                     
#Tuning mtry
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

#Obtain running time.
model_rf_tune$times




