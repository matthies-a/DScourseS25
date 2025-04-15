#errors with neural network and SVM, so I wouldnt bother running those


library(tidyverse)
library(tidymodels)
library(magrittr)
library(modelsummary)
library(rpart)
library(e1071)
library(kknn)
library(nnet)
library(kernlab)
library(parsnip)
library(rsample)
library(glmnet)

set.seed(100)

income <- read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", col_names = FALSE)
names(income) <- c("age","workclass","fnlwgt","education","education.num","marital.status","occupation","relationship","race","sex","capital.gain","capital.loss","hours","native.country","high.earner")



# From UC Irvine's website (http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names)
#   age: continuous.
#   workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
#   fnlwgt: continuous.
#   education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
#   education-num: continuous.
#   marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
#   occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
#   relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
#   race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
#   sex: Female, Male.
#   capital-gain: continuous.
#   capital-loss: continuous.
#   hours-per-week: continuous.
#   native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

######################
# Clean up the data
######################
# Drop unnecessary columns
income %<>% select(-native.country, -fnlwgt, education.num)
# Make sure continuous variables are formatted as numeric
income %<>% mutate(across(c(age,hours,education.num,capital.gain,capital.loss), as.numeric))
# Make sure discrete variables are formatted as factors
income %<>% mutate(across(c(high.earner,education,marital.status,race,workclass,occupation,relationship,sex), as.factor))
# Combine levels of factor variables that currently have too many levels
income %<>% mutate(education = fct_collapse(education,
                                            Advanced    = c("Masters","Doctorate","Prof-school"), 
                                            Bachelors   = c("Bachelors"), 
                                            SomeCollege = c("Some-college","Assoc-acdm","Assoc-voc"),
                                            HSgrad      = c("HS-grad","12th"),
                                            HSdrop      = c("11th","9th","7th-8th","1st-4th","10th","5th-6th","Preschool") 
),
marital.status = fct_collapse(marital.status,
                              Married      = c("Married-civ-spouse","Married-spouse-absent","Married-AF-spouse"), 
                              Divorced     = c("Divorced","Separated"), 
                              Widowed      = c("Widowed"), 
                              NeverMarried = c("Never-married")
), 
race = fct_collapse(race,
                    White = c("White"), 
                    Black = c("Black"), 
                    Asian = c("Asian-Pac-Islander"), 
                    Other = c("Other","Amer-Indian-Eskimo")
), 
workclass = fct_collapse(workclass,
                         Private = c("Private"), 
                         SelfEmp = c("Self-emp-not-inc","Self-emp-inc"), 
                         Gov     = c("Federal-gov","Local-gov","State-gov"), 
                         Other   = c("Without-pay","Never-worked","?")
), 
occupation = fct_collapse(occupation,
                          BlueCollar  = c("?","Craft-repair","Farming-fishing","Handlers-cleaners","Machine-op-inspct","Transport-moving"), 
                          WhiteCollar = c("Adm-clerical","Exec-managerial","Prof-specialty","Sales","Tech-support"), 
                          Services    = c("Armed-Forces","Other-service","Priv-house-serv","Protective-serv")
)
)


######################
# tidymodels time!
######################
income_split <- initial_split(income, prop = 0.8)
income_train <- training(income_split)
income_test  <- testing(income_split)



library(dials)
library(workflows)
library(tune)
#####################
# logistic regression
#####################
print('Starting LOGIT')
# set up the task and the engine
tune_logit_spec <- logistic_reg(
  penalty = tune(), # tuning parameter
  mixture = 1       # 1 = lasso, 0 = ridge
) %>% 
  set_engine("glmnet") %>%
  set_mode("classification")

# define a grid over which to try different values of the regularization parameter lambda
lambda_grid <- grid_regular(penalty(), levels = 50)

# 3-fold cross-validation
rec_folds <- vfold_cv(income_train, v = 3)

# Workflow
rec_wf <- workflow() %>%
  add_model(tune_logit_spec) %>%
  add_formula(high.earner ~ education + marital.status + race + workclass + occupation + relationship + sex + age + capital.gain + capital.loss + hours)

# Tuning results
rec_res <- rec_wf %>%
  tune_grid(
    resamples = rec_folds,
    grid = lambda_grid
  )

# what is the best value of lambda?
top_acc  <- show_best(rec_res, metric = "accuracy")
best_acc <- select_best(rec_res, metric = "accuracy")
final_logit_lasso <- finalize_workflow(rec_wf,
                                       best_acc
)
print('*********** LOGISTIC REGRESSION **************')
logit_test <- last_fit(final_logit_lasso,income_split) %>%
  collect_metrics()

logit_test %>% print(n = 1)
top_acc %>% print(n = 1)

# combine results into a nice tibble (for later use)
logit_ans <- top_acc %>% slice(1)
logit_ans %<>% left_join(logit_test %>% slice(1),by=c(".metric",".estimator")) %>%
  mutate(alg = "logit") %>% select(-starts_with(".config"))



library(recipes)
library(yardstick)
#####################
# tree model
#####################
print('Starting TREE')

# set up the task and the engine
tune_tree_spec <- decision_tree(
  min_n = tune(), # tuning parameter
  tree_depth = tune(), # tuning parameter
  cost_complexity = tune(), # tuning parameter
) %>% 
  set_engine("rpart") %>%
  set_mode("classification")

# define a set over which to try different values of the regularization parameter (complexity, depth, etc.)
tree_parm_df1 <- tibble(cost_complexity = seq(.001,.2,by=.05))
tree_parm_df2 <- tibble(min_n = seq(10,100,by=10))
tree_parm_df3 <- tibble(tree_depth = seq(5,20,by=5))
tree_parm_df  <- full_join(tree_parm_df1,tree_parm_df2,by=character()) %>% full_join(.,tree_parm_df3,by=character())

# YOU FILL IN THE REST


# Create the workflow (assuming you have a training dataset named 'train_data' with outcome variable 'outcome')
tree_wf <- workflow() %>%
  add_model(tune_tree_spec) %>%
  add_formula(high.earner ~ .) # Replace 'outcome' with your actual target variable name

# Set up cross-validation
set.seed(123) # for reproducibility
cv_folds <- vfold_cv(income_train, v = 3)

# Tune the model across the parameter grid
tree_res <- tree_wf %>%
  tune_grid(
    resamples = cv_folds,
    grid = tree_parm_df,
    metrics = metric_set(accuracy, roc_auc, sens, spec)
  )

# View the top performing models
tree_res %>%
  collect_metrics() %>%
  arrange(desc(mean))

# Get the best hyperparameters
best_tree <- tree_res %>%
  select_best()  # or select_best("roc_auc") depending on your preference

# Finalize the workflow with the best parameters
final_tree_wf <- tree_wf %>%
  finalize_workflow(best_tree)

# Fit the final model to the training data
final_tree_fit <- final_tree_wf %>%
  fit(income_train)

# Make predictions on test data (assuming you have a test dataset named 'test_data')
tree_preds <- final_tree_fit %>%
  predict(income_test) %>%
  bind_cols(income_test %>% select(high.earner))

# Evaluate the model on test data
test_accuracy <- tree_preds %>%
  accuracy(truth = high.earner, estimate = .pred_class)

# Create confusion matrix
conf_mat <- tree_preds %>%
  conf_mat(truth = high.earner, estimate = .pred_class)
#see accuracy
print(conf_mat)
tree_ans <- print(conf_mat)

#####################
# neural net
#####################
print('Starting NNET')
# set up the task and the engine
tune_nnet_spec <- mlp(
  hidden_units = tune(), # tuning parameter
  penalty = tune()
) %>% 
  set_engine("nnet") %>%
  set_mode("classification")

# define a set over which to try different values of the regularization parameter (number of neighbors)
nnet_parm_df1 <- tibble(hidden_units = seq(1,10))
lambda_grid   <- grid_regular(penalty(), levels = 10)
nnet_parm_df  <- full_join(nnet_parm_df1,lambda_grid,by=character())

# YOU FILL IN THE REST

set.seed(234)
income_folds <- vfold_cv(income_train, v = 3, strata = high.earner)

# Create a simple workflow with the preprocessed data
nnet_workflow <- workflow() %>%
  add_model(tune_nnet_spec) %>%
  add_formula(high.earner ~ .)

# Tune the model with grid search
set.seed(345)
nnet_tuning <- tune_grid(
  nnet_workflow,
  resamples = income_folds,
  grid = nnet_parm_df,
  metrics = metric_set(accuracy, roc_auc)
)

# Show tuning results
print(nnet_tuning)
print(collect_metrics(nnet_tuning))

# Visualize tuning results
autoplot(nnet_tuning)

# Select the best hyperparameters
best_params <- select_best(nnet_tuning, metric = "accuracy")
print("Best parameters:")
print(best_params)

# Finalize the workflow with the best parameters
final_workflow <- finalize_workflow(
  nnet_workflow,
  best_params
)

# Fit the final model on the entire training set
final_fit <- fit(final_workflow, data = iris_train)

# Evaluate on the test set
test_predictions <- predict(final_fit, new_data = iris_test)
test_probabilities <- predict(final_fit, new_data = iris_test, type = "prob")

# Combine predictions with actual values
test_results <- bind_cols(
  test_predictions,
  test_probabilities,
  iris_test %>% select(Species)
)

# Calculate accuracy
test_accuracy <- accuracy(test_results, truth = Species, estimate = .pred_class)
print("Test accuracy:")
print(test_accuracy)

# Create confusion matrix
conf_mat <- conf_mat(test_results, truth = Species, estimate = .pred_class)
print("Confusion Matrix:")
print(conf_mat)
autoplot(conf_mat)

# ROC curve and AUC
test_roc <- test_results %>%
  roc_curve(Species, .pred_setosa:.pred_virginica) 

print("ROC Curve:")
autoplot(test_roc)

test_auc <- test_results %>%
  roc_auc(Species, .pred_setosa:.pred_virginica)

print("AUC:")
print(test_auc)

# Additional model evaluation metrics
classification_metrics <- test_results %>%
  metrics(truth = Species, estimate = .pred_class)
print("Additional classification metrics:")
print(classification_metrics)

# Generate predictions for a specific example
# Replace with your actual prediction scenario
new_data_example <- iris_test[1:5, ]
predictions <- predict(final_fit, new_data = new_data_example)
prediction_probs <- predict(final_fit, new_data = new_data_example, type = "prob")
print("Example predictions:")
print(bind_cols(new_data_example, predictions, prediction_probs))

# Save the model
saveRDS(final_fit, "nnet_final_model.rds")

# Optional: Extract and display the model
final_model <- extract_fit_parsnip(final_fit)
print(summary(final_model$fit))

print('NNET Complete')






#####################
# knn
#####################
print('Starting KNN')
# set up the task and the engine
tune_knn_spec <- nearest_neighbor(
  neighbors = tune() # tuning parameter
) %>% 
  set_engine("kknn") %>%
  set_mode("classification")

# define a set over which to try different values of the regularization parameter (number of neighbors)
knn_parm_df <- tibble(neighbors = seq(1,30))

# YOU FILL IN THE REST


# Create cross-validation folds
set.seed(234)
cv_folds <- vfold_cv(income_train, v = 3)

# Create workflow
knn_workflow <- workflow() %>%
  add_model(tune_knn_spec) %>%
  add_formula(high.earner ~ .)  # Adjust formula based on your predictors

# Tune the model
knn_tune_results <- tune_grid(
  knn_workflow,
  resamples = cv_folds,
  grid = knn_parm_df,
  metrics = metric_set(accuracy, roc_auc)
)

# Select best model
best_knn <- select_best(knn_tune_results, metric = "accuracy")

# Finalize workflow with best parameters
final_knn_wf <- finalize_workflow(knn_workflow, best_knn)

# Fit final model on training data
final_knn_fit <- fit(final_knn_wf, income_train)

# Predict on test data
test_predictions <- predict(final_knn_fit, income_test) %>%
  bind_cols(income_test)

# Evaluate performance
test_metrics <- test_predictions %>%
  metrics(truth = class, estimate = .pred_class)

print(test_metrics)




#####################
# SVM
#####################
print('Starting SVM')
# set up the task and the engine
tune_svm_spec <- svm_rbf(
  cost = tune(), 
  rbf_sigma = tune()
) %>% 
  set_engine("kernlab") %>%
  set_mode("classification")

# define a set over which to try different values of the regularization parameter (number of neighbors)
svm_parm_df1 <- tibble(cost      = c(2^(-2),2^(-1),2^0,2^1,2^2,2^10))
svm_parm_df2 <- tibble(rbf_sigma = c(2^(-2),2^(-1),2^0,2^1,2^2,2^10))
svm_parm_df  <- full_join(svm_parm_df1,svm_parm_df2,by=character())

# YOU FILL IN THE REST

# Set up tuning grid
svm_grid <- grid_regular(
  cost(range = c(-2, 10), trans = log2_trans()),
  rbf_sigma(range = c(-2, 10), trans = log2_trans()),
  levels = 5
)

# Create workflow
svm_workflow <- workflow() %>%
  add_model(tune_svm_spec) %>%
  add_formula(high.earner ~ .)

# Tune the model
set.seed(234)
svm_tune <- tune_grid(
  svm_workflow,
  resamples = cv_folds,
  grid = svm_grid,
  metrics = metric_set(accuracy, roc_auc),
  control = control_grid(save_pred = TRUE)
)

# Show best parameters
best_params <- select_best(svm_tune, metric = "accuracy")
print("Best parameters:")
print(best_params)

# Finalize the model with best parameters
final_svm_spec <- finalize_model(tune_svm_spec, best_params)
final_workflow <- workflow() %>%
  add_model(final_svm_spec) %>%
  add_formula(high.earner ~ .)

# Fit final model
final_fit <- final_workflow %>%
  fit(data = income_train)

# Predict on test set
test_predictions_svm <- predict(final_fit, income_test) %>%
  bind_cols(income_test)

# Evaluate performance
test_metrics_svm <- test_predictions_svm %>%
  metrics(truth = Species, estimate = .pred_class)
print("Test set performance:")
print(test_metrics_svm)

# Optional: Visualize results
autoplot(svm_tune)



#####################
# combine answers
#####################
all_ans <- bind_rows(logit_ans,tree_ans)
datasummary_df(all_ans %>% select(-.metric,-.estimator,-mean,-n,-std_err),output="markdown") %>% print

