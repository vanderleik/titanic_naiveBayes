# Titanic: Machine Learning from Disaster > versão 2

setwd("c:/FCD/PortfolioR/titanic2")
getwd()

# title: 'Titanic: Machine Learning from Disaster - Naive Bayes'
# author: 'Vanderlei Kleinschmidt'
# date: '07 december 2020'

# Pré processamento
source("prepro_titanic.R")

# packages
library(naivebayes)
library(caret)

# Para treinar o Naive Bayes precisamos nos certificar que os dados estão com os tipos corretos. O algorítmo requer que a variável dependente seja do tipo factor ou character.
sapply(train, class)
head(train, 10)

# Treinando o modelo Naive Bayes
nb_poisson <- naive_bayes(train$Survived ~ ., train, laplace = 0, usekernel = FALSE, usepoisson = TRUE)

# Uma vez que o modelo foi treinado, preciso verificar qual a capacidade preditiva dele. 
# Para isso eu crio um objeto chamado 'previsao_p', uso a função predict, passando como parâmetro o modelo
# treinado e os dados de treino.

previsao_p = predict(nb_poisson, newdata = train[-1])

# Crio a matriz de confução para avaliar a capacidade preditiva do modelo
matriz = table(train$Survived, previsao_p)

# A matriz de confusão é criada usando a função 'confusionMatrix' do pacote 'caret'
confusionMatrix(matriz)

# Sumarizando o resultado do modelo treinado
summary(nb_poisson)

# Agora que o modelo foi treinado, posso usá-lo nos dados de teste para depois gerar o arquivo e enviar ao Kagle
modelo_p <- predict(nb_poisson, test, type = "class")

# Gerando o arquivo e enviado ao Kagle
submission_p <- data.frame(PassengerId = passengers, Survived = modelo_p)
write.csv(submission_p,'titanic_nb_p.csv', row.names = FALSE)
