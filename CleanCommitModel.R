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
  add_model(rf_mod)

test_preds <- predict(final_wf, data, type = "prob") %>%
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

## Tuning RF Model
#####

### Train/test split (leave test untouched)

split <- initial_split(data, prop = 0.8, strata = BYU)
trainData <- training(split)
testData  <- testing(split)

trainData <- trainData %>%
  mutate(BYU = factor(BYU, levels = c("Y","N")))

testData <- testData %>%
  mutate(BYU = factor(BYU, levels = c("Y","N")))

folds <- vfold_cv(trainData, v = 5, strata = BYU)

### Recipe with SMOTE (done correctly)

rec <- recipe(BYU ~ ., data = trainData) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(BYU, neighbors = 5)

### Model: Random Forest

rf_mod <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = tune()
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

### Workflow

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_mod)

### Tuning grid

grid <- grid_regular(
  mtry(range = c(2, 8)),
  min_n(),
  trees(range = c(300, 800)),
  levels = 5
)

### Cross-validation (correct ROC AUC)

cv_results <- wf %>%
  tune_grid(
    resamples = folds,
    grid = grid,
    metrics = metric_set(roc_auc)
  )

show_best(cv_results, metric = "roc_auc")

### Fit final model

best_params <- select_best(cv_results, metric = "roc_auc")

### mtry = 8, trees = 425, min_n = 40

final_wf <- wf %>%
  finalize_workflow(best_params) %>%
  fit(trainData)

### Predict on your real test set

test_preds <- predict(final_wf, testData, type = "prob") %>%
  bind_cols(testData %>% select(BYU))

roc_auc(test_preds, truth = BYU, .pred_Y) # 0.838

#####

## Final RF Model
#####
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
  add_model(rf_mod)

test_preds <- predict(final_wf, data, type = "prob") %>%
  bind_cols(data %>% select(BYU))

## ROC_AUC - 0.9238838

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