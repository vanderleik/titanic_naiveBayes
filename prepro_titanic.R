# Titanic: Machine Learning from Disaster > versão 2

setwd("c:/FCD/PortfolioR/titanic2")
getwd()

# title: 'Titanic: Machine Learning from Disaster'
# author: 'Vanderlei Kleinschmidt'
# date: '17 october 2020'

# packages
library(tidyverse)
# library(patchwork)
# library(class)
# library(gmodels)

#1. Introdução

#2. Coletando os dados
train <- read.csv('train.csv', stringsAsFactors = F)
test  <- read.csv('test.csv', stringsAsFactors = F)

# Vai ser útil no final
survived <- train$Survived
survived <- as.factor(survived)
passengers <- test$PassengerId

# Agrupando e analisando os dados

train$isTrainSet <- TRUE
test$isTrainSet <- FALSE

test$Survived <- NA

fulldata <- rbind(train, test)

# Cada classe tem idades diferentes, sendo que na primeira classe a média das idades é maior do que nas demais, e a terceira classe é a que tem a menor média de idade.
ageMedian_1 <- fulldata %>% group_by(Pclass) %>% filter(Pclass == 1)
median_1 <- median(ageMedian_1$Age, na.rm = TRUE)

ageMedian_2 <- fulldata %>% group_by(Pclass) %>% filter(Pclass == 2)
median_2 <- median(ageMedian_2$Age, na.rm = TRUE)

ageMedian_3 <- fulldata %>% group_by(Pclass) %>% filter(Pclass == 3)
median_3 <- median(ageMedian_3$Age, na.rm = TRUE)

rotulos <- c("1a classe", "2a classe", "3a classe")
mediana <- c(median_1, median_2, median_3)
medianas <- rbind(rotulos, mediana)

impute_age <- function(age, class){
  out <- age
  for (i in 1:length(age)){
    
    if (is.na(age[i])){
      
      if (class[i] == 1){
        out[i] <- median_1
        
      }else if (class[i] == 2){
        out[i] <- median_2
        
      }else{
        out[i] <- median_3
      }
    }else{
      out[i]<-age[i]
    }
  }
  return(out)
}

fixed.ages <- impute_age(fulldata$Age, fulldata$Pclass)
fulldata$Age <- fixed.ages

fareMedian_1 <- fulldata %>% group_by(Pclass) %>% filter(Pclass == 1)
median_f1 <- median(fareMedian_1$Fare)

fareMedian_2 <- fulldata %>% group_by(Pclass) %>% filter(Pclass == 2)
median_f2 <- median(fareMedian_2$Fare)

fareMedian_3 <- fulldata %>% group_by(Pclass) %>% filter(Pclass == 3)
median_f3 <- median(fareMedian_3$Fare)

rotulosf <- c("1a classe", "2a classe", "3a classe")
medianaf <- c(median_f1, median_f2, median_f3)
medianasf <- rbind(rotulosf, medianaf)

median_f3 <- median(fareMedian_3$Fare, na.rm = TRUE)

fulldata[is.na(fulldata$Fare), "Fare"] <- median_f3

fulldata[fulldata$Embarked == "", "Embarked"] <- 'S'
fulldata[fulldata$Sex == "female", "Female"] <- '1'
fulldata[fulldata$Sex == "male", "Female"] <- '0'
fulldata[fulldata$Pclass == "1", "FClasse"] <- '1'
fulldata[fulldata$Pclass == "2", "FClasse"] <- '0'
fulldata[fulldata$Pclass == "3", "FClasse"] <- '0'
fulldata[fulldata$Pclass == "1", "SClasse"] <- '0'
fulldata[fulldata$Pclass == "2", "SClasse"] <- '1'
fulldata[fulldata$Pclass == "3", "SClasse"] <- '0'
fulldata[fulldata$Pclass == "1", "TClasse"] <- '0'
fulldata[fulldata$Pclass == "2", "TClasse"] <- '0'
fulldata[fulldata$Pclass == "3", "TClasse"] <- '1'
fulldata[fulldata$Embarked == "C", "CEmbarked"] <- '1'
fulldata[fulldata$Embarked == "Q", "CEmbarked"] <- '0'
fulldata[fulldata$Embarked == "S", "CEmbarked"] <- '0'
fulldata[fulldata$Embarked == "C", "QEmbarked"] <- '0'
fulldata[fulldata$Embarked == "Q", "QEmbarked"] <- '1'
fulldata[fulldata$Embarked == "S", "QEmbarked"] <- '0'
fulldata[fulldata$Embarked == "C", "SEmbarked"] <- '0'
fulldata[fulldata$Embarked == "Q", "SEmbarked"] <- '0'
fulldata[fulldata$Embarked == "S", "SEmbarked"] <- '1'

fulldata$Survived <- as.factor(fulldata$Survived)
fulldata$Sex <- as.factor(fulldata$Sex)
fulldata$Female <-  as.factor(fulldata$Female)
fulldata$Pclass <- as.factor(fulldata$Pclass)
fulldata$FClasse <- as.factor(fulldata$FClasse)
fulldata$SClasse <- as.factor(fulldata$SClasse)
fulldata$TClasse <- as.factor(fulldata$TClasse)
fulldata$Embarked <- as.factor(fulldata$Embarked)
fulldata$CEmbarked <- as.factor(fulldata$CEmbarked)
fulldata$QEmbarked <- as.factor(fulldata$QEmbarked)
fulldata$SEmbarked <- as.factor(fulldata$SEmbarked)

sobreviventes <- table(train$Survived) # Utilizo os dados de treino porque não tenho a informação de quantos sobreviveram nos dados de teste.

gender <- table(fulldata$Sex)
gender_train <- table(train$Sex)
gender_test <- table(test$Sex)
bySex <- with(train, table(Survived, Sex)) # Dados de treino porque não sabemos quantos sobreviveram nos dados de teste.

SocEconClass <- table(fulldata$Pclass)

byClass <- with(train, table(Survived, Pclass)) # Utilizo os dados de treino porque não tenho a informação de quantos sobreviveram nos dados de teste.

#4. Escolhendo o algoritmo de machine learning e treinando o modelo com os dados

train$Female <- NA
test$Female <- NA

train$FClasse <- NA
test$FClasse <- NA
train$SClasse <- NA
test$SClasse <- NA
train$TClasse <- NA
test$TClasse <- NA

train$CEmbarked <- NA
test$CEmbarked <- NA
train$QEmbarked <- NA
test$QEmbarked <- NA
train$SEmbarked <- NA
test$SEmbarked <- NA

train <- fulldata[fulldata$isTrainSet == TRUE,]
test <- fulldata[fulldata$isTrainSet == FALSE,]

train <- select(train, -Sex, -Pclass, -TClasse, -Embarked, -SEmbarked, -PassengerId, -Name, -Ticket, -Cabin, -isTrainSet)
test <- select(test, -Sex, -Pclass, -TClasse, -Embarked, -SEmbarked, -PassengerId, -Name, -Ticket, -Cabin, -isTrainSet)

#4.1 Probabilistic Learning – Classification Using Naive Bayes

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

train$Age <- normalize(train$Age)
train$SibSp <- normalize(train$SibSp)
train$Parch <- normalize(train$Parch)
train$Fare <- normalize(train$Fare)

# Faço o mesmo procedimento para os dados de teste
test$Age <- normalize(test$Age)
test$SibSp <- normalize(test$SibSp)
test$Parch <- normalize(test$Parch)
test$Fare <- normalize(test$Fare)

print('Fim do pre processamento')