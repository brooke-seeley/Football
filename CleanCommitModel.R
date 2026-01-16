library(tidyverse)
library(tidymodels)
library(readxl)
library(embed)
library(vroom)
library(yardstick)
library(themis)

set.seed(123)

## Original Model
#####

## Read in Data

data <- read_excel("RecruitmentPrediction.xlsx", sheet = "Data") %>%
  select(
    EThree, Position, Utah, Distance, Height, Weight,
    Score, LDS, Alumni, Poly, BYU
  ) %>%
  mutate(
    across(c(EThree, Utah, LDS, Alumni, Poly), as.factor),
    Position = as.factor(Position),
    BYU = factor(BYU, levels = c("N", "Y"))  # 'Y' = positive class
  )

## Recipe with SMOTE

rec <- recipe(BYU ~ ., data = data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(BYU, neighbors = 5)

## Random Forest Model

rf_mod <- rand_forest(
  mtry  = 8,
  min_n = 40,
  trees = 675
  ) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_mod) %>%
  fit(data)

test_preds <- predict(wf, data, type = "prob") %>%
  bind_cols(data %>% select(BYU))

## ROC_AUC - 0.9090533

full_roc <- roc_auc(test_preds, truth = BYU, .pred_Y)
roc_auc <- 1 - full_roc$.estimate
roc_auc

## Accuracy - 0.8616352

full_preds <- read_excel("byu_commitment_model.xlsx")

test_results <- full_preds %>%
  mutate(
    Predicted = factor(if_else(Commitment_Probability >= 0.5, 
                               "Committed", "Not Committed"), 
                       levels = c("Committed", "Not Committed"))
  ) %>%
  mutate(Actual = factor(Actual))

accuracy_table <- accuracy(test_results, truth = Actual, estimate = Predicted)
accuracy <- accuracy_table$.estimate
accuracy

#####

## Changing to 247
#####
## Read in Data

data <- read_excel("RecruitmentPrediction.xlsx", sheet = "247Data") %>%
  select(
    '247Top', Position, Utah, Distance, Height, Weight,
    Score, LDS, Alumni, Poly, BYU
  ) %>%
  mutate(
    across(c('247Top', Utah, LDS, Alumni, Poly), as.factor),
    Position = as.factor(Position),
    BYU = factor(BYU, levels = c("N", "Y"))  # 'Y' = positive class
  )

## Recipe with SMOTE

rec <- recipe(BYU ~ ., data = data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(BYU, neighbors = 5)

## Random Forest Model

rf_mod <- rand_forest(
  mtry  = 8,
  min_n = 40,
  trees = 425
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_mod) %>%
  fit(data)

test_preds <- predict(wf, data, type = "prob") %>%
  bind_cols(data %>% select(BYU))

## ROC_AUC - 0.9473439

full_roc <- roc_auc(test_preds, truth = BYU, .pred_Y)
roc_auc <- 1 - full_roc$.estimate
roc_auc

## Accuracy - 0.8805031

full_preds <- read_excel("byu_247_model.xlsx")

test_results <- full_preds %>%
  mutate(
    Predicted = factor(if_else(Commitment_Probability >= 0.5, 
                               "Committed", "Not Committed"), 
                       levels = c("Committed", "Not Committed"))
  ) %>%
  mutate(Actual = factor(Actual))

accuracy_table <- accuracy(test_results, truth = Actual, estimate = Predicted)
accuracy <- accuracy_table$.estimate
accuracy
#####