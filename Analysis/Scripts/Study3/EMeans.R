library(tidyverse)  # for read_csv and mutate
library(lme4)       # for lmer
library(emmeans)    # for emmeans and pairs

df_long <- read_csv('/Users/sm6511/Desktop/Prediction-Accomodation-Exp/Analysis/Study3.0/df_long_for_R.csv') |>
  mutate(
    task = relevel(factor(task), ref = "predict"),
    feature_relevance = relevel(factor(feature_relevance), ref = "irrelevant"),
    feature_dimension = relevel(factor(feature_dimension), ref = "feet")
  )
df_long$feature_relevance <- relevel(factor(df_long$feature_relevance), ref = "irrelevant")
df_long$feature_dimension <- relevel(factor(df_long$feature_dimension), ref = "feet")

# Fit model
model <- lmer(feature_importance ~ task * feature_relevance * feature_dimension + 
                (1|participant), data = df_long)

# emmeans
emmeans(model, ~ task | feature_relevance * feature_dimension) |>
  pairs(adjust = "bonferroni")

