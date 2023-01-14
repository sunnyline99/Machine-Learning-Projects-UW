## Load packages

library(tidyverse)
library(tree)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(here)
library(purrr)
library(tidyr)
library(ggplot2)
library(MASS)
library(pROC)
library(e1071)

## Set wd and seed to reproduce

setwd("C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Classification")
set.seed(123456789)

## Data downloaded from - https://www.kaggle.com/code/aniketgupta1001/hit-song-prediction-using-random-forest-ensemble/notebook
## Load data and take a look at the structure

spotify <- read.csv("spotify.csv")

spotify <- spotify %>% as_tibble()
glimpse(spotify)

## delete track_id and track_title (maybe they may be useful to merge some other data)

spotify <- spotify [,-c(1,3)]

## Recode 0 and 1 to factors

spotify$On_chart <- factor(spotify$On_chart, levels = c(0, 1), labels = c('No', 'Yes'))

## Split data to test and train samples

training_obs <- createDataPartition(spotify$On_chart, 
                                    p = 0.7, 
                                    list = FALSE)
spotify.train <- spotify[training_obs,]
spotify.test  <- spotify[-training_obs,]

## Take a look at variables distributions

spotify.train %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

table(spotify.train$On_chart)

## Here create correlation matrix

## Maybe recode artist name to some groups -> maybe PCA or sth like that??


## Save both datasets as rds files

spotify.train %>% saveRDS(here("C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Classification", "spotify.train.rds"))
spotify.test  %>% saveRDS(here("C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Classification", "spotify.test.rds"))


## Here we need to perform data cleaning for both datasets












## First model - without artist name as it was making some problems

model1.formula <- On_chart ~ duration_ms + energy + key + mode + 
  time_signature + acousticness + danceability + instrumentalness + liveness + 
  speechiness + valence + tempo 

spotify.tree1 <- 
  rpart(model1.formula,
        data = spotify.train,
        method = "class")

spotify.tree1
summary(spotify.tree1)

rpart.plot(spotify.tree1)
fancyRpartPlot(spotify.tree1)

summary(spotify.tree1)


## 2nd model - same results

spotify.tree2 <- 
  rpart(model1.formula,
        data = spotify.train,
        method = "class",
        minsplit = 1000, 
        minbucket = 500,
        maxdepth = 10)

spotify.tree2
fancyRpartPlot(spotify.tree2)


## 2nd model without restrictions - same results

spotify.tree2a <- 
  rpart(model1.formula,
        data = spotify.train,
        method = "class",
        minsplit = 1000, 
        minbucket = 500,
        maxdepth = 10,
        # we don't impose any restriction on the tree growth 
        cp = -1)
fancyRpartPlot(spotify.tree2)


## 3rd model - pruning 

spotify.tree3 <- 
  rpart(model1.formula,
        data = spotify.train,
        method = "class",
        minsplit = 500, # ~ 2% of the training set
        minbucket = 250, # ~ 1% of the training set
        maxdepth = 30, # default
        cp = -1)
fancyRpartPlot(spotify.tree3)
spotify.tree3

pred.tree3 <- predict(spotify.tree3, spotify.train, type = "class")
head(pred.tree3)

## Confusion Matrix

confusionMatrix(data = pred.tree3, # predictions
                # actual values
                reference = spotify.train$On_chart,
                # definitions of the "success" label
                positive = "Yes")

printcp(spotify.tree3)

opt <- which.min(spotify.tree3$cptable[, "xerror"])
opt

cp <- spotify.tree3$cptable[opt, "CP"]
cp

## Actual pruning 

spotify.tree3p <- 
  prune(spotify.tree3, cp = cp)
fancyRpartPlot(spotify.tree3p)

## ROC Curve

pred.train.tree3  <- predict(spotify.tree3,  spotify.train)
pred.train.tree3p <- predict(spotify.tree3p, spotify.train)

ROC.train.tree3  <- roc(as.numeric(spotify.train$On_chart == "Yes"), 
                        pred.train.tree3[, 1])
ROC.train.tree3p <- roc(as.numeric(spotify.train$On_chart == "Yes"), 
                        pred.train.tree3p[, 1])


list(
  ROC.train.tree3  = ROC.train.tree3,
  ROC.train.tree3p = ROC.train.tree3p
) %>%
  pROC::ggroc(alpha = 0.5, linetype = 1, size = 1) + 
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color = "grey", 
               linetype = "dashed") +
  labs(subtitle = paste0("Gini TRAIN: ",
                         "tree3 = ", 
                         round(100*(2 * auc(ROC.train.tree3) - 1), 1), "%, ",
                         "tree3p = ", 
                         round(100*(2 * auc(ROC.train.tree3p) - 1), 1), "% ")) +
  theme_bw() + coord_fixed() +
  # scale_color_brewer(palette = "Paired") +
  scale_color_manual(values = RColorBrewer::brewer.pal(n = 4, 
                                                       name = "Paired")[c(1, 3)])


## Same for the testing set

pred.test.tree3  <- predict(spotify.tree3, 
                            spotify.test)
pred.test.tree3p <- predict(spotify.tree3p, 
                            spotify.test)
ROC.test.tree3  <- roc(as.numeric(spotify.test$On_chart == "Yes"), 
                       pred.test.tree3[, 1])
ROC.test.tree3p <- roc(as.numeric(spotify.test$On_chart == "Yes"), 
                       pred.test.tree3p[, 1])


list(
  ROC.train.tree3  = ROC.train.tree3,
  ROC.test.tree3   = ROC.test.tree3,
  ROC.train.tree3p = ROC.train.tree3p,
  ROC.test.tree3p  = ROC.test.tree3p
) %>%
  pROC::ggroc(alpha = 0.5, linetype = 1, size = 1) + 
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color = "grey", 
               linetype = "dashed") +
  labs(subtitle = paste0("Gini TRAIN: ",
                         "tree3 = ", 
                         round(100*(2 * auc(ROC.train.tree3) - 1), 1), "%, ",
                         "tree3p = ", 
                         round(100*(2 * auc(ROC.train.tree3p) - 1), 1), "% ",
                         "Gini TEST: ",
                         "tree3 = ", 
                         round(100*(2 * auc(ROC.test.tree3) - 1), 1), "%, ",
                         "tree3p = ", 
                         round(100*(2 * auc(ROC.test.tree3p) - 1), 1), "% "
  )) +
  theme_bw() + coord_fixed() +
  scale_color_brewer(palette = "Paired")


## Ended on 2nd classes classification subject - row 538
## finish from the 2nd classess






## 3rd classes - bagging & random forrest


## 4th XGBoost for classification

## 5th ensembling models - maybe ensemble after Neural Network

## 6th Neural Networks

## It should be enough to pass 
