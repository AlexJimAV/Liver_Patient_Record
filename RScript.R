## Alejandro Jiménez
## Indian Liver Patient Record Project - HarvardX: PH125.9x
## May 23, 2020

######################## Load dataset ########################

#Install packages (if necessary) and load them
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(neuralnet)) install.packages("caret", repos = "http://cran.us.r-project.org")

#Indian Liver Patient Records dataset:
  # https://www.kaggle.com/uciml/indian-liver-patient-records?select=indian_liver_patient.csv
  # Data previously downloaded in a .zip file and extract the dataset.

#Inspecting the file, it seems that it have column names.
#.csv files, .R and .Rmd must be treated as a R project to load correctly the files
#Load dataset
patients <- read_csv("indian_liver_patient.csv")

#See class of the variables
sapply(patients,class)

#Change Dataset into the fist column, change name into liver and make it a factor class.
#Change Gender to second column
patients <- patients[,c(11,1:10)]
patients <- patients[,c(1,3,c(2,4:11))]
patients <- patients %>% mutate(Dataset=as.factor(Dataset)) %>% rename(Liver=Dataset)

#change gender into factors class
patients <- patients %>% mutate(Gender=as.factor((Gender)))

#Check if there are missing values
apply(is.na(patients), 2, which)

#Change missing missing NA into 0
patients <- patients %>% mutate_all(~replace(., is.na(.), 0))

#See class of the variables
sapply(patients,class)

######################## Data Exploration ########################

#Identify which value in liver indicates a desease
summary(patients$Liver)

#According to the information given by the link. There are 416 patients with liver disease.
#Factor 1 indicates a liver disease.

#Age vs liver boxplot with gender distinction
patients %>% ggplot(aes(Liver,Age,fill=Gender))+
  geom_boxplot() + ggtitle("Liver vs Age Boxplot")

#Gender Barplot with liver distiction
patients %>% group_by(Gender) %>% ggplot(aes(Gender,fill=Liver))+
  geom_bar() + ylab("Count")+ggtitle("Barplot of Gender with Disease Distiction")

#Proportion of Liver by Gender
patients %>% group_by(Gender) %>% summarise(Disease=mean(Liver==1),Not_Disease = mean(Liver==2))

#Direct vs Total Bilirubin with liver distiction
patients %>% ggplot(aes(Total_Bilirubin,Direct_Bilirubin,color=Liver))+
  geom_rug()+geom_point() +
  xlab("Total Bilirubin") + ylab("Direct Bilirubin")+
  ggtitle("Total vs Direct Bilirubin")

#Scale features
x <- as.matrix(patients[,3:11])
x_scaled <- sweep(sweep(x,2,colMeans(x)),
                  2, colSds(x),FUN="/")
#Heatmap of features
d_features <- dist(t(x_scaled))
heatmap(as.matrix(d_features),labRow = NA, labCol = NA)

#Principal Component Analysis
pca <- prcomp(x_scaled)
#Boxplot
data.frame(type = patients$Liver ,pca$x) %>%
  gather(key = "PC",value="value", -type) %>% 
  ggplot(aes(PC,value,fill = type)) + geom_boxplot()

######################## Data Analysis without Gender########################

# Creating data partition
set.seed(1,sample.kind = "Rounding")
test_index <- createDataPartition(patients$Liver, time=1, p=0.15,list=FALSE)
test_x <- x_scaled[test_index,]
test_y <- patients$Liver[test_index]
train_x <- x_scaled[-test_index,]
train_y <- patients$Liver[-test_index]

#Define Fuction to predict with K-means
predict_kmeans <- function(k, x){
  centers <- k$centers
  n_centers <- nrow(centers)
  dist_mat <- as.matrix(dist(rbind(centers, x)))
  dist_mat <- dist_mat[-seq(n_centers), seq(n_centers)]
  max.col(-dist_mat)
}

# K-means Clustering
set.seed(1,sample.kind = "Rounding")
k <- kmeans(train_x, centers = 2)
#Predict values
kmeans_pred <- predict_kmeans(k,test_x)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=as.factor(kmeans_pred),reference=test_y)
results <- data.frame(Method = "K-Means",
                      Accuracy = cm$overall["Accuracy"],
                      Sensitivity=cm$byClass["Sensitivity"],
                      Specificity=cm$byClass["Specificity"],
                      Balanced_Accuracy= cm$byClass["Balanced Accuracy"])

#Logistic Regression
train_glm <- train(train_x,train_y,method="glm")
glm_pred <- predict(train_glm,test_x)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=glm_pred,reference=test_y)
results <- bind_rows(results,data.frame(Method = "Logistic Regression (glm)",
                      Accuracy = cm$overall["Accuracy"],
                      Sensitivity=cm$byClass["Sensitivity"],
                      Specificity=cm$byClass["Specificity"],
                      Balanced_Accuracy= cm$byClass["Balanced Accuracy"]))

#LDA
train_lda <- train(train_x,train_y,method="lda")
lda_pred <- predict(train_lda,test_x)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=lda_pred,reference=test_y)
results <- bind_rows(results,data.frame(Method = "LDA",
                                        Accuracy = cm$overall["Accuracy"],
                                        Sensitivity=cm$byClass["Sensitivity"],
                                        Specificity=cm$byClass["Specificity"],
                                        Balanced_Accuracy= cm$byClass["Balanced Accuracy"]))

#QDA
train_qda <- train(train_x,train_y,method="qda")
qda_pred <- predict(train_qda,test_x)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=qda_pred,reference=test_y)
results <- bind_rows(results,data.frame(Method = "QDA",
                                        Accuracy = cm$overall["Accuracy"],
                                        Sensitivity=cm$byClass["Sensitivity"],
                                        Specificity=cm$byClass["Specificity"],
                                        Balanced_Accuracy= cm$byClass["Balanced Accuracy"]))

#Loess
set.seed(1, sample.kind = "Rounding")
train_loess <- train(train_x,train_y,method="gamLoess")
loess_pred <- predict(train_loess,test_x)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=loess_pred,reference=test_y)
results <- bind_rows(results,data.frame(Method = "Loess",
                                        Accuracy = cm$overall["Accuracy"],
                                        Sensitivity=cm$byClass["Sensitivity"],
                                        Specificity=cm$byClass["Specificity"],
                                        Balanced_Accuracy= cm$byClass["Balanced Accuracy"]))

#kNN
set.seed(1, sample.kind = "Rounding")
tuning <- data.frame(k=seq(3,21,2))
train_knn <- train(train_x,train_y,method="knn",tuneGrid = tuning)
knn_pred <- predict(train_knn,test_x)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=knn_pred,reference=test_y)
results <- bind_rows(results,data.frame(Method = "kNN",
                                        Accuracy = cm$overall["Accuracy"],
                                        Sensitivity=cm$byClass["Sensitivity"],
                                        Specificity=cm$byClass["Specificity"],
                                        Balanced_Accuracy= cm$byClass["Balanced Accuracy"]))

# Random Forest
set.seed(1, sample.kind = "Rounding")
tuning <- data.frame(mtry=c(3,5,7,9))
train_rf <- train(train_x,train_y, method="rf",tuneGrid = tuning, importance = TRUE)
rf_pred <- predict(train_rf, test_x)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=rf_pred,reference=test_y)
results <- bind_rows(results,data.frame(Method = "Random Forest (rf)",
                                        Accuracy = cm$overall["Accuracy"],
                                        Sensitivity=cm$byClass["Sensitivity"],
                                        Specificity=cm$byClass["Specificity"],
                                        Balanced_Accuracy= cm$byClass["Balanced Accuracy"]))

#Ensemble 
ensemble <- cbind(glm=glm_pred==1,lda=lda_pred==1,qda=qda_pred==1,loess=loess_pred==1,
                  rf=rf_pred==1,knn=knn_pred==1,kmeans=kmeans_pred==1)
ensemble_pred <- ifelse(rowMeans(ensemble) >0.5,1,2)
cm <- confusionMatrix(data=as.factor(ensemble_pred),reference=test_y)
#Show Results of Accuracy, Sensitivity and Specificity
results <- bind_rows(results,data.frame(Method = "Ensemble",
                                        Accuracy = cm$overall["Accuracy"],
                                        Sensitivity=cm$byClass["Sensitivity"],
                                        Specificity=cm$byClass["Specificity"],
                                        Balanced_Accuracy= cm$byClass["Balanced Accuracy"]))
results %>% knitr::kable()

######################## Data Analysis with Gender ########################
#Add numeric value to gender
patients <- patients %>% mutate(num_gender = 0)
patients$num_gender[which(patients$Gender=="Female")] <- 1
patients$num_gender[which(patients$Gender=="Male")] <- -1

#Scale features
x <- as.matrix(patients[,3:12])
x_scaled <- sweep(sweep(x,2,colMeans(x)),
                  2, colSds(x),FUN="/")

# Creating data partition
set.seed(1,sample.kind = "Rounding")
test_index <- createDataPartition(patients$Liver, time=1, p=0.15,list=FALSE)
test_x <- x_scaled[test_index,]
test_y <- patients$Liver[test_index]
train_x <- x_scaled[-test_index,]
train_y <- patients$Liver[-test_index]

# K-means Clustering

set.seed(1,sample.kind = "Rounding")
k <- kmeans(train_x, centers = 2)
#Predict values
kmeans_pred <- predict_kmeans(k,test_x)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=as.factor(kmeans_pred),reference=test_y)
results_g <- data.frame(Method_Gender = "K-Means",
                      Accuracy = cm$overall["Accuracy"],
                      Sensitivity=cm$byClass["Sensitivity"],
                      Specificity=cm$byClass["Specificity"],
                      Balanced_Accuracy= cm$byClass["Balanced Accuracy"])

#Logistic Regression
train_glm <- train(train_x,train_y,method="glm")
glm_pred <- predict(train_glm,test_x)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=glm_pred,reference=test_y)
results_g <- bind_rows(results_g,data.frame(Method_Gender = "Logistic Regression (glm)",
                                        Accuracy = cm$overall["Accuracy"],
                                        Sensitivity=cm$byClass["Sensitivity"],
                                        Specificity=cm$byClass["Specificity"],
                                        Balanced_Accuracy= cm$byClass["Balanced Accuracy"]))

#LDA
train_lda <- train(train_x,train_y,method="lda")
lda_pred <- predict(train_lda,test_x)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=lda_pred,reference=test_y)
results_g <- bind_rows(results_g,data.frame(Method_Gender = "LDA",
                                        Accuracy = cm$overall["Accuracy"],
                                        Sensitivity=cm$byClass["Sensitivity"],
                                        Specificity=cm$byClass["Specificity"],
                                        Balanced_Accuracy= cm$byClass["Balanced Accuracy"]))

#QDA
train_qda <- train(train_x,train_y,method="qda")
qda_pred <- predict(train_qda,test_x)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=qda_pred,reference=test_y)
results_g <- bind_rows(results_g,data.frame(Method_Gender = "QDA",
                                        Accuracy = cm$overall["Accuracy"],
                                        Sensitivity=cm$byClass["Sensitivity"],
                                        Specificity=cm$byClass["Specificity"],
                                        Balanced_Accuracy= cm$byClass["Balanced Accuracy"]))

#Loess
set.seed(1, sample.kind = "Rounding")
train_loess <- train(train_x,train_y,method="gamLoess")
loess_pred <- predict(train_loess,test_x)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=loess_pred,reference=test_y)
results_g <- bind_rows(results_g,data.frame(Method_Gender = "Loess",
                                        Accuracy = cm$overall["Accuracy"],
                                        Sensitivity=cm$byClass["Sensitivity"],
                                        Specificity=cm$byClass["Specificity"],
                                        Balanced_Accuracy= cm$byClass["Balanced Accuracy"]))

#kNN
set.seed(1, sample.kind = "Rounding")
tuning <- data.frame(k=seq(3,21,2))
train_knn <- train(train_x,train_y,method="knn",tuneGrid = tuning)
knn_pred <- predict(train_knn,test_x)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=knn_pred,reference=test_y)
results_g <- bind_rows(results_g,data.frame(Method_Gender = "kNN",
                                        Accuracy = cm$overall["Accuracy"],
                                        Sensitivity=cm$byClass["Sensitivity"],
                                        Specificity=cm$byClass["Specificity"],
                                        Balanced_Accuracy= cm$byClass["Balanced Accuracy"]))

# Random Forest
set.seed(1, sample.kind = "Rounding")
tuning <- data.frame(mtry=c(3,5,7,9))
train_rf <- train(train_x,train_y, method="rf",tuneGrid = tuning, importance = TRUE)
rf_pred <- predict(train_rf, test_x)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=rf_pred,reference=test_y)
results_g <- bind_rows(results_g,data.frame(Method_Gender = "Random Forest (rf)",
                                        Accuracy = cm$overall["Accuracy"],
                                        Sensitivity=cm$byClass["Sensitivity"],
                                        Specificity=cm$byClass["Specificity"],
                                        Balanced_Accuracy= cm$byClass["Balanced Accuracy"]))

#Ensemble 
ensemble <- cbind(glm=glm_pred==1,lda=lda_pred==1,qda=qda_pred==1,loess=loess_pred==1,
                  rf=rf_pred==1,knn=knn_pred==1,kmeans=kmeans_pred==1)
ensemble_pred <- ifelse(rowMeans(ensemble) >0.5,1,2)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=as.factor(ensemble_pred),reference=test_y)
results_g <- bind_rows(results_g,data.frame(Method_Gender = "Ensemble",
                                        Accuracy = cm$overall["Accuracy"],
                                        Sensitivity=cm$byClass["Sensitivity"],
                                        Specificity=cm$byClass["Specificity"],
                                        Balanced_Accuracy= cm$byClass["Balanced Accuracy"]))
results_g %>% knitr::kable()

######################## Data Analysis with Neural Network ########################

#Bind x_scaled with patients$liver in a dataframe
patients_nn <- cbind(patients$Liver,as.data.frame(x_scaled))
patients_nn <- patients_nn %>% rename(Liver=`patients$Liver`)

#Change values of liver into binary and numeric
patients_nn <- patients_nn %>% mutate(Liver=as.numeric(Liver))
patients_nn$Liver[which(patients_nn$Liver==2)] <- 0

# Creating data partition
set.seed(1,sample.kind = "Rounding")
test_index <- createDataPartition(patients_nn$Liver, time=1, p=0.15,list=FALSE)
test <- as.data.frame(patients_nn[test_index,])
train <- as.data.frame(patients_nn[-test_index,])

#Neural Network method
nn <- neuralnet(Liver~Age+Total_Bilirubin+Direct_Bilirubin+Alkaline_Phosphotase+
               Alamine_Aminotransferase+Aspartate_Aminotransferase+
               Total_Protiens+Albumin+
               Albumin_and_Globulin_Ratio+num_gender,
               data=train, hidden=c(10,5),act.fct = "logistic",linear.output = FALSE)

#Plot the neural network
plot(nn)

#Predict Values with Neural Netwokr
nn_pred <- compute(nn,test[,2:11])
nn_pred <- ifelse(nn_pred$net.result<0.5,1,0)
#Show Results of Accuracy, Sensitivity and Specificity
cm <- confusionMatrix(data=as.factor(nn_pred),reference=as.factor(test[,1]))
results <- bind_rows(results,data.frame(Method = "Neural Network",
                                            Accuracy = cm$overall["Accuracy"],
                                            Sensitivity=cm$byClass["Sensitivity"],
                                            Specificity=cm$byClass["Specificity"],
                                            Balanced_Accuracy= cm$byClass["Balanced Accuracy"]))

######################## Results ########################

results %>% knitr::kable()
results_g %>% knitr::kable()
