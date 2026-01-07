library(tidyverse)
library(tidymodels)
library(readxl)
library(embed)
library(vroom)
library(yardstick)

## Initial Data Reads
#####

### Train Data (before 2026)
# 
# trainData <- read_excel("RecruitmentPrediction.xlsx", sheet = "Data") %>%
#   filter(Class != '2026') %>%
#   select(-First, -Last, -Class) %>%
#   mutate(BYU = case_when(
#     BYU == "N" ~ 0,
#     BYU == "Y" ~ 1)) %>%
#   mutate(BYU = factor(BYU))
# 
# trainDataClean <- trainData %>%
#   mutate(across(c(EThree, Utah, LDS, Alumni, Poly),
#                 ~ ifelse(. == "Y", 1, 0)))
# 
# vroom_write(x=trainDataClean, file="./traindata.csv", delim=',')
# 
# trainDataUnclean <- read_excel("RecruitmentPrediction.xlsx", sheet = "Data") %>%
#   filter(Class != '2026') %>%
#   mutate(BYU = case_when(
#     BYU == "N" ~ 0,
#     BYU == "Y" ~ 1)) %>%
#   select(-Class)
# 
# vroom_write(x=trainDataUnclean, file="./newtraindata.csv", delim=',')
# 
# ### Test Data (2026)
# 
# testData <- read_excel("RecruitmentPrediction.xlsx", sheet = "Data") %>%
#   filter(Class == '2026') %>%
#   select(-First, -Last, -Class, -BYU)
# 
# testDataClean <- testData %>%
#   mutate(across(c(EThree, Utah, LDS, Alumni, Poly),
#                 ~ ifelse(. == "Y", 1, 0)))
# 
# vroom_write(x=testDataClean, file="./testdata.csv", delim=',')
# 
# testDataUnclean <- read_excel("RecruitmentPrediction.xlsx", sheet = "Data") %>%
#   filter(Class == '2026') %>%
#   select(-First, -Last)
# 
# vroom_write(x=testDataUnclean, file="./newtestdata.csv", delim=',')
# 
# ### Actual 2026 Response
# 
# test_labels <- read_excel("RecruitmentPrediction.xlsx", sheet = "Data") %>%
#   filter(Class == 2026) %>% 
#   select(BYU) %>%
#   mutate(BYU = case_when(
#     BYU == "N" ~ 0,
#     BYU == "Y" ~ 1)) %>%
#   mutate(BYU = factor(BYU))

#####

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

# library(glmnet)
# 
# log_reg_model <- logistic_reg() %>%
#   set_engine("glm")
# 
# log_reg_workflow <- workflow() %>%
#   add_recipe(recipe) %>%
#   add_model(log_reg_model) %>%
#   fit(data=trainData)
# 
# ### Predictions
# 
# log_reg_predictions <- predict(log_reg_workflow,
#                                new_data=testData,
#                                type="prob")
# 
# ### Calculate roc_auc
# 
# test_results <- log_reg_predictions %>%
#   bind_cols(tibble(truth = test_labels$BYU))
# 
# roc_auc(test_results, truth = truth, .pred_1)

#####

## roc_auc was less than 0.5 because it was predicting 0 more, trying balancing
## in the recipe and then other models

## Recipe w/ SMOTE
#####

# library(themis)
# 
# smote_recipe <- recipe(BYU ~ ., data = trainData) %>%
#   step_mutate(across(c(EThree, Utah, LDS, Alumni, Poly),
#                      ~ ifelse(. == "Y", 1, 0))) %>%
#   step_mutate_at(Position, fn = factor) %>%
#   step_dummy(Position) %>%
#   step_smote(all_outcomes(), neighbors=1)
# 
# smote_prep <- prep(smote_recipe)
# bake(smote_prep, new_data = NULL)

#####

## Upsampling
#####

# library(themis)
# 
# up_recipe <- recipe(BYU ~ ., data = trainData) %>%
#   step_mutate(across(c(EThree, Utah, LDS, Alumni, Poly),
#                      ~ ifelse(. == "Y", 1, 0))) %>%
#   step_mutate_at(Position, fn = factor) %>%
#   step_dummy(Position) %>%
#   step_upsample(all_outcomes())
# 
# up_prep <- prep(up_recipe)
# bake(up_prep, new_data = NULL)

#####

## Downsampling
#####

# library(themis)
# 
# down_recipe <- recipe(BYU ~ ., data = trainData) %>%
#   step_mutate(across(c(EThree, Utah, LDS, Alumni, Poly),
#                      ~ ifelse(. == "Y", 1, 0))) %>%
#   step_mutate_at(Position, fn = factor) %>%
#   step_dummy(Position) %>%
#   step_downsample(all_outcomes())
# 
# down_prep <- prep(down_recipe)
# bake(down_prep, new_data = NULL)

#####

## Penalized Logistic Regression - roc_auc: varies, but still < 0.3 w/ SMOTE,
## w/ Upsample: 0.167, w/ Downsample: 0.175
#####

# library(glmnet)
# 
# preg_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>%
#   set_engine("glmnet")
# 
# preg_workflow <- workflow() %>%
#   add_recipe(down_recipe) %>%
#   add_model(preg_mod)
# 
# ### Grid of values to tune over
# 
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 5)
# 
# ### Split data for CV
# 
# folds <- vfold_cv(trainData, v = 5, repeats = 1)
# 
# ### Run the CV
# 
# CV_results <- preg_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics(metric_set(roc_auc)))
# 
# ### Find Best Tuning Parameters
# 
# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")
# print(bestTune)
# 
# ### Finalize the Workflow & fit it
# 
# final_wf <-
#   preg_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=trainData)
# 
# ### Predict
# 
# pen_reg_predictions <- final_wf %>%
#   predict(new_data = testData, type="prob")
# 
# ### Calculate roc_auc
# 
# test_results <- pen_reg_predictions %>%
#   bind_cols(tibble(truth = test_labels$BYU))
# 
# roc_auc(test_results, truth = truth, .pred_1)

#####

## Trying Regression Trees w/ Upsample - roc_auc: 0.156
#####

# library(rpart)
# 
# tree_mod <- rand_forest(mtry=tune(),
#                         min_n=tune(),
#                         trees=tune()) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# tree_workflow <- workflow() %>%
#   add_recipe(up_recipe) %>%
#   add_model(tree_mod)
# 
# ### Grid of values
# 
# tuning_grid <- grid_regular(mtry(range=c(1,9)),
#                             min_n(),
#                             trees(range=c(100,1000)),
#                             levels=5)
# 
# ### CV
# 
# folds <- vfold_cv(trainData, v = 5, repeats = 1)
# 
# CV_results <- tree_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=(metric_set(roc_auc)))
# 
# ### Find best tuning parameters
# 
# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")
# 
# ### Finalize workflow
# 
# final_wf <-
#   tree_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=trainData)
# 
# ### Predict
# 
# tree_predictions <- final_wf %>%
#   predict(new_data = testData, type="prob")
# 
# ### Calculate roc_auc
# 
# test_results <- tree_predictions %>%
#   bind_cols(tibble(truth = test_labels$BYU))
# 
# roc_auc(test_results, truth = truth, .pred_1)

#####

## DataRobot 
#####

# ### Elastic-Net Classifier (mixing alpha=0.5 / Binomial Deviance) - 0.325
# 
# elasticnetresults <- vroom('elasticnetresult.csv')
# 
# test_results <- elasticnetresults %>%
#   bind_cols(tibble(truth = test_labels$BYU))
# 
# roc_auc(test_results, truth = truth, BYU_PREDICTION)
# 
# ## Making a Model Using All Data (no test)
# 
# ### Prep for DataRobot
# 
# data_full <- read_excel("RecruitmentPrediction.xlsx", sheet = "Data") %>%
#   mutate(BYU = case_when(
#     BYU == "N" ~ 0,
#     BYU == "Y" ~ 1)) %>%
#   select(-Class)
# 
# vroom_write(x=data_full, file="./data_full.csv", delim=',')
# 
# ### Predictions - Light Gradient Boosting on Elastic Net Predictions
# 
# no_resp <- read_excel("RecruitmentPrediction.xlsx", sheet = "Data") %>%
#   select(-Class, -BYU)
# 
# vroom_write(x=no_resp, file="./no_resp.csv", delim=',')
# 
# full_preds <- read.csv('full_preds.csv', header = TRUE)
# 
# ### Comparison
# 
# compare <- data.frame(data_full$BYU, 
#                       full_preds$BYU_PREDICTION, full_preds$BYU_1_PREDICTION)
# 
# df <- tibble(
#   truth = factor(data_full$BYU),
#   pred  = factor(full_preds$BYU_PREDICTION)
# )
# 
# accuracy(df, truth = truth, estimate = pred)
# 
# mean(data_full$BYU == 0)
# 
# results <- tibble(
#   truth = factor(data_full$BYU),
#   .pred_1 = full_preds$BYU_1_PREDICTION
# )
# 
# roc_auc(results, truth = truth, .pred_1)
# 
# p <- full_preds$BYU_1_PREDICTION
# y <- data_full$BYU
# pROC::roc(y, 1 - p)$auc

#####

## Making Models Based on Features DataRobot Called Important
#####

# ### Read in Data
# 
# trainData <- read_excel("RecruitmentPrediction.xlsx", sheet = "Data") %>%
#   mutate(BYU = case_when(
#     BYU == "N" ~ 0,
#     BYU == "Y" ~ 1)) %>%
#   mutate(BYU = factor(BYU)) %>%
#   select(EThree, Utah, Score, LDS, Alumni, BYU)
# 
# set.seed(123)
# split <- initial_split(trainData, prop = 0.8, strata = BYU)
# trainData <- training(split)
# testData <- testing(split)
# 
# ### Upsampling on Train Data because of Imbalance
# 
# majority <- trainData %>% filter(BYU == 0)
# minority <- trainData %>% filter(BYU == 1)
#  
# minority_upsampled <- minority %>% slice_sample(n = nrow(majority), 
#                                                 replace = TRUE)
# 
# trainData <- bind_rows(majority, minority_upsampled)
# 
# ### Recipe
# 
# recipe <- recipe(BYU ~ ., data = trainData) %>%
#   step_mutate(across(c(EThree, Utah, LDS, Alumni), ~ factor(.))) %>%
#   step_dummy(all_factor_predictors()) %>%
#   step_interact(terms = ~ Utah_Y:LDS_Y + LDS_Y:Alumni_Y + EThree_Y:Score)
# 
# recipe_prep <- prep(recipe)
# bake(recipe_prep, new_data = trainData)

#####

### Random Forest Model - roc_auc: 0.218
#####

# library(rpart)
# 
# tree_mod <- rand_forest(mtry=tune(),
#                         min_n=tune(),
#                         trees=tune()) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# tree_workflow <- workflow() %>%
#   add_recipe(recipe) %>%
#   add_model(tree_mod)
# 
# ### Grid of values
# 
# tuning_grid <- grid_regular(mtry(range=c(1,5)),
#                             min_n(),
#                             trees(range=c(100,1000)),
#                             levels=5)
# 
# ### CV
# 
# folds <- vfold_cv(trainData, v = 5, repeats = 1)
# 
# CV_results <- tree_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=(metric_set(roc_auc)))
# 
# ### Find best tuning parameters
# 
# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")
# 
# ### Finalize workflow
# 
# final_wf <-
#   tree_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=trainData)
# 
# ### Predict
# 
# tree_predictions <- final_wf %>%
#   predict(new_data = testData, type="prob")
# 
# ### Calculate roc_auc
# 
# test_results <- tree_predictions %>%
#   bind_cols(testData %>% select(BYU))
# 
# roc_auc(test_results, truth = BYU, .pred_1)

#####

### Penalized Logistic Regression - roc_auc: 0.0901
#####

# library(glmnet)
# 
# preg_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>%
#   set_engine("glmnet")
# 
# preg_workflow <- workflow() %>%
#   add_recipe(recipe) %>%
#   add_model(preg_mod)
# 
# ### Grid of values to tune over
# 
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 5)
# 
# ### CV
# 
# folds <- vfold_cv(trainData, v = 5, repeats = 1)
# 
# CV_results <- preg_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics(metric_set(roc_auc)))
# 
# ### Find Best Tuning Parameters
# 
# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")
# print(bestTune)
# 
# ### Finalize the Workflow & fit it
# 
# final_wf <-
#   preg_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=trainData)
# 
# ### Predict
# 
# pen_reg_predictions <- final_wf %>%
#   predict(new_data = testData, type="prob")
# 
# ### Calculate roc_auc
# 
# test_results <- pen_reg_predictions %>%
#   bind_cols(testData %>% select(BYU))
# 
# roc_auc(test_results, truth = BYU, .pred_1)

#####

## Yikes, both super low. Manual upsampling may not be the way.

## Suggested RF Model from ChatGPT - 0.895, it looks like my model WAS good,
## R was just misunderstanding levels. So, technically, all of my roc_aucs
## should be flipped.
#####

### Clean setup (this fixes factor levels AND probability direction)

library(themis)

set.seed(123)

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

### mtry = 8, trees = 675, min_n = 40

final_wf <- wf %>%
  finalize_workflow(best_params) %>%
  fit(trainData)

### Predict on your real test set

test_preds <- predict(final_wf, testData, type = "prob") %>%
  bind_cols(testData %>% select(BYU))

roc_auc(test_preds, truth = BYU, .pred_Y)

#####

## Prep for Presenting
#####

library(readr)

### Get predicted probabilities on test data

full_preds <- final_wf %>%
  predict(new_data = data, type = "prob") %>%
  bind_cols(data)

### Clean column names for Tableau

tableau_data <- full_preds %>%
  mutate(
    Actual = if_else(BYU == "Y", "Committed", "Not Committed")
  ) %>%
  select(
    Actual,
    Commitment_Probability = .pred_Y,
    EThree, Position, Utah, Distance, Height, Weight, Score, LDS, Alumni, Poly
  )

### Export for Tableau

library(writexl)

write_xlsx(tableau_data, "byu_commitment_model.xlsx")

### A Little Look-See

full_roc <- roc_auc(full_preds, truth = BYU, .pred_Y)

1 - full_roc$.estimate

### Accuracy-Wise

full_preds <- read_excel("byu_commitment_model.xlsx")

test_results <- full_preds %>%
  mutate(
    Predicted = factor(if_else(Commitment_Probability >= 0.5, 
                               "Committed", "Not Committed"), 
                       levels = c("Committed", "Not Committed"))
  ) %>%
  mutate(Actual = factor(Actual))

# Compute accuracy
accuracy(test_results, truth = Actual, estimate = Predicted)

#####