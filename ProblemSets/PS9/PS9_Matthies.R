library(tidymodels)
library(glmnet)
library(rsample)
library(recipes)
library(caret)

#load data
housing <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", header = FALSE)
names(housing) <- c("crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", 
                    "rad", "tax", "ptratio", "b", "lstat", "medv")

#split data into training and test
set.seed(123456)
housing_split <- initial_split(housing, prop = 0.8)
housing_train <- training(housing_split)
housing_test  <- testing(housing_split)



housing_recipe <- recipe ( medv ~ . , data = housing ) %>%
  # convert outcome variable to logs
  step_log( all_outcomes ()) %>%
  # convert 0/1 chas to a factor
  step_bin2factor(chas) %>%
  # create interaction term between crime and nox
  step_interact( terms = ~ crim : zn : indus :rm: age : rad : tax :
                      ptratio : b : lstat : dis : nox ) %>%
  # create square terms of some continuous variables
  step_poly(crim , zn , indus ,rm , age , rad , tax , ptratio ,b ,
                lstat , dis , nox , degree =6)
# Run the recipe
housing_prep <- housing_recipe %>% prep (housing_train , retain
                                               = TRUE )
housing_train_prepped <- housing_prep %>% juice
housing_test_prepped <- housing_prep %>% bake (new_data = housing_test)
# create x and y training and test data
housing_train_x <- housing_train_prepped %>% select ( - medv )
housing_test_x <- housing_test_prepped %>% select ( - medv )
housing_train_y <- housing_train_prepped %>% select ( medv )
housing_test_y <- housing_test_prepped %>% select ( medv )

dim(housing_train)
#Dimension of training data is 404x14
#Original data is 506x14

#LASSO
library(parsnip)

lasso_spec <- linear_reg(penalty=0.5,mixture=1) %>%
  set_engine("glmnet") %>%  
  set_mode("regression") 

lasso_fit <- lasso_spec %>%
  fit(medv ~ ., data=housing_train_prepped)

ols_spec <- linear_reg() %>%       # Specify a model
  set_engine("lm") %>%   # Specify an engine: lm, glmnet, stan, keras, spark
  set_mode("regression")



housing$log_medv <- log(housing$medv) #log which is in housing_train_y but that was
#throwing me off

# Create the predictors matrix and response vector
# Remove the response variables from the predictors
x_vars <- housing %>% 
  select(-medv, -log_medv) %>%
  as.matrix()
y_var <- housing$log_medv

# Set up 6-fold cross-validation
set.seed(123)  # For reproducibility
folds <- createFolds(y_var, k = 6, list = FALSE)

# Generate sequence of lambda values
lambda_seq <- 10^seq(1, -3, length.out = 100)

# Train LASSO model with cross-validation
cv_lasso <- cv.glmnet(x_vars, y_var, alpha = 1, lambda = lambda_seq, 
                      nfolds = 6, foldid = folds)

# Find the optimal lambda
optimal_lambda <- cv_lasso$lambda.min
cat("Optimal lambda:", optimal_lambda, "\n")
# optimal lambda w/ 6-fold CV states .001



# predict RMSE in sample
library(yardstick)
library(Metrics)
lasso_fit %>% predict(housing_train_prepped) %>%
  mutate(truth = housing_train_prepped$medv) %>%
  rmse(truth,`.pred`) %>%
  print
#RMSE in sample is 0.413

# predict RMSE out of sample
lasso_fit %>% predict(housing_test_prepped) %>%
  mutate(truth = housing_test_prepped$medv) %>%
  rmse(truth,`.pred`) %>%
  print
#RMSE out-of-sample is 0.390



#RIDGE

# Take the log of the median house value for both training and test data
housing_train$log_medv <- log(housing_train$medv)
housing_test$log_medv <- log(housing_test$medv)

# Create the predictors matrix and response vector for training data
x_train <- housing_train %>% 
  select(-medv, -log_medv) %>%
  as.matrix()
y_train <- housing_train$log_medv

# Create the predictors matrix and response vector for test data
x_test <- housing_test %>% 
  select(-medv, -log_medv) %>%
  as.matrix()
y_test <- housing_test$log_medv

# Set up 6-fold cross-validation
set.seed(123)  # For reproducibility
folds <- createFolds(y_train, k = 6, list = FALSE)

# Generate sequence of lambda values
lambda_seq <- 10^seq(3, -3, length.out = 100)

# Train RIDGE model with cross-validation (alpha = 0 for ridge regression)
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0, lambda = lambda_seq, 
                      nfolds = 6, foldid = folds)

# Find the optimal lambda
optimal_lambda <- cv_ridge$lambda.min
cat("Optimal lambda for ridge regression:", optimal_lambda, "\n")
#optimal lambda states 0.00464


# Plot the cross-validation results
plot(cv_ridge)

# Fit the final model with the optimal lambda
final_ridge_model <- glmnet(x_train, y_train, alpha = 0, lambda = optimal_lambda)

# Make predictions on the test data
y_pred <- predict(final_ridge_model, newx = x_test, s = optimal_lambda)

# Calculate RMSE on test data
rmse_test <- sqrt(mean((y_test - y_pred)^2))
cat("Out-of-sample RMSE:", rmse_test, "\n")
#out-of-sample RMSE is 0.167