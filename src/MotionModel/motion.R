library(neuralnet)
library(caret)

# pass in neural net model object
nntext <- function(m) {
  all_weights <- m$weights[[1]]
  i <- 1
  for(theta_i in all_weights) {
    weights_i <- all_weights[[i]]
    weights_i_t <- c(t(weights_i))

    for(weight_i in weights_i_t) {
      cat(sprintf("%.12f ", weight_i))
    }

    cat('\n')
    i <- i + 1
  }
}

# load training examples from CSV
motion <- read.csv('motion_examples.csv', stringsAsFactors=TRUE)

# randomly divide into training set & test set (40% of full set set aside for test/evaluation purposes, 60% for training)
set.seed(123)
total_obs_count <- dim(motion)[1]
test_set_count <- (total_obs_count * 0.4)
motion_samples <- sample(total_obs_count, total_obs_count - test_set_count)
motion_train <- motion[motion_samples, ]
motion_test <- motion[-motion_samples, ]

# distribution of vertical/non-vertical training examples in training set & test set
round(prop.table(table(motion_train$orientation)) * 100, digits = 2)
round(prop.table(table(motion_test$orientation)) * 100, digits = 2)

# train neural network
mmodel <- neuralnet(
    orientation ~ roll + yaw + pitch + qw + qx + qy + qz,
    data = motion_train,
    hidden = c(28,21,14,7),
    err.fct = "sse",
    act.fct = "logistic")

# try predicting against test set
motion_result <- compute(mmodel, motion_test[-8])
motion_pred <- motion_result$net.result

# correlation / confusion matrix to evaluate performance
cor_motion <- cor(motion_pred, motion_test$orientation)
confusionMatrix(round(motion_pred), motion_test$orientation, dnn=c('Predicted', 'Actual'))
