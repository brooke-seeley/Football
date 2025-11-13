library(tidyverse)
library(tidymodels)
library(readxl)
library(embed)
library(vroom)
library(yardstick)

## Read in Data

### Train Data (before 2026)

trainData <- read_excel("RecruitmentPrediction.xlsx", sheet = "Data") %>%
  filter(Class != '2026') %>%
  select(-First, -Last, -Class) %>%
  mutate(BYU = case_when(
    BYU == "N" ~ 0,
    BYU == "Y" ~ 1)) %>%
  mutate(BYU = factor(BYU))

trainDataClean <- trainData %>%
  mutate(across(c(EThree, Utah, LDS, Alumni, Poly),
                ~ ifelse(. == "Y", 1, 0)))

vroom_write(x=trainDataClean, file="./traindata.csv", delim=',')

trainDataUnclean <- read_excel("RecruitmentPrediction.xlsx", sheet = "Data") %>%
  filter(Class != '2026') %>%
  mutate(BYU = case_when(
    BYU == "N" ~ 0,
    BYU == "Y" ~ 1))

vroom_write(x=trainDataUnclean, file="./newtraindata.csv", delim=',')

### Test Data (2026)

testData <- read_excel("RecruitmentPrediction.xlsx", sheet = "Data") %>%
  filter(Class == '2026') %>%
  select(-First, -Last, -Class, -BYU)

testDataClean <- testData %>%
  mutate(across(c(EThree, Utah, LDS, Alumni, Poly),
                ~ ifelse(. == "Y", 1, 0)))

vroom_write(x=testDataClean, file="./testdata.csv", delim=',')

testDataUnclean <- read_excel("RecruitmentPrediction.xlsx", sheet = "Data") %>%
  filter(Class == '2026') %>%
  mutate(BYU = case_when(
    BYU == "N" ~ 0,
    BYU == "Y" ~ 1))

vroom_write(x=testDataUnclean, file="./newtestdata.csv", delim=',')

### Actual 2026 Response

test_labels <- read_excel("RecruitmentPrediction.xlsx", sheet = "Data") %>%
  filter(Class == 2026) %>% 
  select(BYU) %>%
  mutate(BYU = case_when(
    BYU == "N" ~ 0,
    BYU == "Y" ~ 1)) %>%
  mutate(BYU = factor(BYU))

## Initial Recipe
#####

# recipe <- recipe(BYU ~ ., data = trainData) %>%
#   step_mutate(across(c(EThree, Utah, LDS, Alumni, Poly),
#                      ~ factor(ifelse(. == "Y", 1, 0)))) %>%
#   step_mutate_at(Position, fn = factor) %>%
#   step_lencode_mixed(Position, outcome = vars(BYU)) %>%
#   step_normalize(all_numeric_predictors())
# 
# recipe_prep <- prep(recipe)
# bake(recipe_prep, new_data = trainData)

#####

## Logistic Regression Model - roc_auc: 0.181
#####

library(glmnet)

log_reg_model <- logistic_reg() %>%
  set_engine("glm")

log_reg_workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(log_reg_model) %>%
  fit(data=trainData)

### Predictions

log_reg_predictions <- predict(log_reg_workflow,
                               new_data=testData,
                               type="prob")

### Calculate roc_auc

test_results <- log_reg_predictions %>%
  bind_cols(tibble(truth = test_labels$BYU))

roc_auc(test_results, truth = truth, .pred_1)

#####

## roc_auc was less than 0.5 because it was predicting 0 more, trying balancing
## in the recipe and then other models

## Recipe w/ SMOTE
#####

library(themis)

smote_recipe <- recipe(BYU ~ ., data = trainData) %>%
  step_mutate(across(c(EThree, Utah, LDS, Alumni, Poly),
                     ~ ifelse(. == "Y", 1, 0))) %>%
  step_mutate_at(Position, fn = factor) %>%
  step_dummy(Position) %>%
  step_smote(all_outcomes(), neighbors=1)

smote_prep <- prep(smote_recipe)
bake(smote_prep, new_data = NULL)

#####

## Upsampling
#####

library(themis)

up_recipe <- recipe(BYU ~ ., data = trainData) %>%
  step_mutate(across(c(EThree, Utah, LDS, Alumni, Poly),
                     ~ ifelse(. == "Y", 1, 0))) %>%
  step_mutate_at(Position, fn = factor) %>%
  step_dummy(Position) %>%
  step_upsample(all_outcomes())

up_prep <- prep(up_recipe)
bake(up_prep, new_data = NULL)

#####

## Downsampling
#####

library(themis)

down_recipe <- recipe(BYU ~ ., data = trainData) %>%
  step_mutate(across(c(EThree, Utah, LDS, Alumni, Poly),
                     ~ ifelse(. == "Y", 1, 0))) %>%
  step_mutate_at(Position, fn = factor) %>%
  step_dummy(Position) %>%
  step_downsample(all_outcomes())

down_prep <- prep(down_recipe)
bake(down_prep, new_data = NULL)

#####

## Penalized Logistic Regression - roc_auc: varies, but still < 0.3 w/ SMOTE,
## w/ Upsample: 0.167, w/ Downsample: 0.175
#####

library(glmnet)

preg_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>%
  set_engine("glmnet")

preg_workflow <- workflow() %>%
  add_recipe(down_recipe) %>%
  add_model(preg_mod)

### Grid of values to tune over

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

### Split data for CV

folds <- vfold_cv(trainData, v = 5, repeats = 1)

### Run the CV

CV_results <- preg_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics(metric_set(roc_auc)))

### Find Best Tuning Parameters

bestTune <- CV_results %>%
  select_best(metric="roc_auc")
print(bestTune)

### Finalize the Workflow & fit it

final_wf <-
  preg_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainData)

### Predict

pen_reg_predictions <- final_wf %>%
  predict(new_data = testData, type="prob")

### Calculate roc_auc

test_results <- pen_reg_predictions %>%
  bind_cols(tibble(truth = test_labels$BYU))

roc_auc(test_results, truth = truth, .pred_1)

#####

## Trying Regression Trees w/ Upsample - roc_auc: 0.156
#####

library(rpart)

tree_mod <- rand_forest(mtry=tune(),
                        min_n=tune(),
                        trees=tune()) %>%
  set_engine("ranger") %>%
  set_mode("classification")

tree_workflow <- workflow() %>%
  add_recipe(up_recipe) %>%
  add_model(tree_mod)

### Grid of values

tuning_grid <- grid_regular(mtry(range=c(1,9)),
                            min_n(),
                            trees(range=c(100,1000)),
                            levels=5)

### CV

folds <- vfold_cv(trainData, v = 5, repeats = 1)

CV_results <- tree_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=(metric_set(roc_auc)))

### Find best tuning parameters

bestTune <- CV_results %>%
  select_best(metric="roc_auc")

### Finalize workflow

final_wf <-
  tree_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainData)

### Predict

tree_predictions <- final_wf %>%
  predict(new_data = testData, type="prob")

### Calculate roc_auc

test_results <- tree_predictions %>%
  bind_cols(tibble(truth = test_labels$BYU))

roc_auc(test_results, truth = truth, .pred_1)

#####

## DataRobot 
### Elastic-Net Classifier (mixing alpha=0.5 / Binomial Deviance) - 0.325

elasticnetresults <- vroom('elasticnetresult.csv')

test_results <- elasticnetresults %>%
  bind_cols(tibble(truth = test_labels$BYU))

roc_auc(test_results, truth = truth, BYU_PREDICTION)

###