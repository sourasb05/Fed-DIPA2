library(nnet)
library(rstatix)
library(dplyr)
data <- read.csv(file = './annotation_wise_regression_table.csv')
data <- data %>% convert_as_factor(gender, nationality)
formula <- "ifPrivacy ~ gender + age + nationality + extraversion + agreeableness + conscientiousness + neuroticism + openness"
sum(data$ifPrivacy == 0)
sum(data$ifPrivacy == 1)
log.model <- glm(formula = formula, family = "binomial", data = data, weights = ifelse(data$ifPrivacy == 1, sum(data$ifPrivacy == 0) / sum(data$ifPrivacy == 1), 1))
summary(log.model)
summary(log.model)$coefficients
#odd ratio: The larger the odds ratio, the more likely the event is to be found with exposure. The smaller the odds ratio is than 1, the less likely the event is to be found with exposure.
exp(coefficients(log.model))
# 95% CI
confint.default(log.model)
AIC(log.model)
