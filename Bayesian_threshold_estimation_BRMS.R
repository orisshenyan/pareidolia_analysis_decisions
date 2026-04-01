library(tidyverse)
library(brms)

set.seed(10402026)

results <- training2 %>%
  group_by(subnumber) %>%
  group_map(~ {
    subj <- .y$subnumber
    df <- .x
    
    cat("Fitting:", subj, "\n")
    
    # --- Frequentist ---
    fit_freq <- tryCatch(
      glm(Correct_Response ~ Opacity, family = binomial, data = df),
      error = function(e) NULL
    )
    if (is.null(fit_freq)) return(NULL)
    
    thresh_freq <- (qlogis(0.65) - coef(fit_freq)[1]) / coef(fit_freq)[2]
    
    # --- Bayesian ---
    nl_formula <- bf(
      Correct_Response ~ inv_logit(beta * (Opacity - threshold) + 0.6190392), #logit(0.65)
      threshold ~ 1,
      beta ~ 1,
      nl = TRUE
    )
    
    my_priors <- c(
      prior(normal(0.264, 0.085), nlpar = "threshold"), #from the pilot data
      prior(normal(7.6, 3), nlpar = "beta", lb = 0) #also from the pilot data
    )
    
    fit_bayes <- tryCatch(
      brm(
        nl_formula,
        data = df,
        family = bernoulli(link = "identity"),
        prior = my_priors,
        chains = 4,
        iter = 4000,
        control = list(adapt_delta = 0.99),
        init = 0,
        silent = 2,
        refresh = 0
      ),
      error = function(e) NULL
    )
    if (is.null(fit_bayes)) return(NULL)
    
    post <- as_draws_df(fit_bayes)
    
    tibble(
      subnumber = subj,
      threshold_freq = as.numeric(thresh_freq),
      threshold_bayes = median(post$b_threshold_Intercept),
      bayes_lower = quantile(post$b_threshold_Intercept, 0.025),
      bayes_upper = quantile(post$b_threshold_Intercept, 0.975),
      n_trials = nrow(df)
    )
  }) %>%
  bind_rows()

write.csv(results, "Data/threshold_estimates_BRMS.csv", row.names = FALSE)