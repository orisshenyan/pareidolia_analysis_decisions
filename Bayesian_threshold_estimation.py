# ===================================================================
# Bayesian psychometric threshold estimation — aligned with brms
# ===================================================================

import pandas as pd
import glob
import numpy as np
import pymc as pm
import arviz as az
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

# ===================================================================
# Prior settings (matching brms)
# ===================================================================
prior_mean_threshold = 0.264
prior_sd_threshold   = 0.085
prior_mean_beta      = 7.6       #
prior_sd_beta        = 3.0       # matching brms prior(normal(7.6, 3))
target               = 0.65
logit_target         = np.log(target / (1 - target))  # 0.6190392
randomseed           = 31032026

# ===================================================================
# Import all subjects
# ===================================================================
subject_dirs = [
    d for d in glob.glob("Data/individual_data/*/")
    if os.path.isdir(d)
]
subject_dirs.sort()

results = []

# ===================================================================
# Main loop
# ===================================================================
for subject_dir in subject_dirs:

    subject_id = os.path.basename(os.path.normpath(subject_dir))
    print(f"\nProcessing: {subject_id}")

    # ---------------------------------------------------------------
    # 1. Load training block CSVs
    # ---------------------------------------------------------------
    block_files = glob.glob(f"{subject_dir}{subject_id}*_trainingblock*.csv")
    block_files.sort()

    if not block_files:
        print(f"  No block files found for {subject_id}, skipping.")
        continue

    all_blocks = pd.concat([pd.read_csv(f) for f in block_files])
    all_blocks = all_blocks.reset_index(drop=True)

    # ---------------------------------------------------------------
    # 2. Score accuracy
    # ---------------------------------------------------------------
    all_blocks['Correct_Response'] = (
        ((all_blocks['Direction_Report'] == 1) & (all_blocks['Face_Position'] < 0)) |
        ((all_blocks['Direction_Report'] == 2) & (all_blocks['Face_Position'] > 0))
    ).astype(int)

    # ---------------------------------------------------------------
    # 3. Prepare trial-level data
    # ---------------------------------------------------------------
    # CHANGE: use trial-level arrays for Bernoulli likelihood
    #         (instead of aggregated Binomial counts)
    opacity_trials = all_blocks['Opacity'].values.astype(float)
    correct_trials = all_blocks['Correct_Response'].values.astype(int)

    # ---------------------------------------------------------------
    # 4. Bayesian model — aligned with brms
    # ---------------------------------------------------------------
    #
    # brms model:
    #   Correct_Response ~ inv_logit(beta*(Opacity - threshold) + logit(0.65))
    #   threshold ~ Normal(0.264, 0.085)
    #   beta      ~ Normal(7.6, 3), lb = 0
    #   family    = bernoulli(link = "identity")
    #   target_accept  0.99 to match brms
    #   tune=2000, draws=2000 to match brms iter=4000
    # ---------------------------------------------------------------
    with pm.Model() as model:

        # --- Priors (matching brms) ---
        threshold = pm.Normal(
            'threshold',
            mu=prior_mean_threshold,
            sigma=prior_sd_threshold
        )

        # estimate slope with lower-bounded normal prior
        beta = pm.TruncatedNormal(
            'beta',
            mu=prior_mean_beta,
            sigma=prior_sd_beta,
            lower=0            # lb = 0 in brms
        )

        # Cthreshold of 65
        p_correct = pm.math.sigmoid(
            beta * (opacity_trials - threshold) + logit_target
        )

        # Bernoulli trial-level likelihood
        obs = pm.Bernoulli(
            'obs',
            p=p_correct,
            observed=correct_trials
        )

        # draws/tune/target aligned with brms
        trace = pm.sample(
            draws=2000,               # post-warmup draws (brms: iter/2)
            tune=2000,                # warmup draws      (brms: iter/2)
            chains=4,
            target_accept=0.99,       
            return_inferencedata=True,
            progressbar=True,
            random_seed=randomseed,
            initvals={'threshold': 0, 'beta': 1}  # approx. brms init=0
        )

    # ---------------------------------------------------------------
    # 5. Posterior summary — aligned with brms
    # ---------------------------------------------------------------
    # CHANGE: threshold samples are ALREADY the 65% point
    #         (no post-hoc shift needed)
    samples_threshold = trace.posterior['threshold'].values.flatten()

    # CHANGE: use median (not mean) and quantile-based CI (not HDI)
    threshold_bayes = np.median(samples_threshold)
    bayes_lower     = np.quantile(samples_threshold, 0.025)
    bayes_upper     = np.quantile(samples_threshold, 0.975)

    # ---------------------------------------------------------------
    # 6. Frequentist GLM (unchanged)
    # ---------------------------------------------------------------
    glm_model = smf.glm(
        'Correct_Response ~ Opacity',
        data=all_blocks,
        family=sm.families.Binomial()
    ).fit()

    glm_intercept    = glm_model.params['Intercept']
    glm_slope        = glm_model.params['Opacity']
    glm_threshold_65 = (logit_target - glm_intercept) / glm_slope

    # ---------------------------------------------------------------
    # 7. Store results
    # ---------------------------------------------------------------
    results.append({
        'subject':          subject_id,
        'threshold_freq':   glm_threshold_65,
        'threshold_bayes':  threshold_bayes,
        'bayes_lower':      bayes_lower,
        'bayes_upper':      bayes_upper,
        'n_trials':         len(all_blocks),
    })

    print(f"  GLM threshold:       {glm_threshold_65:.3f}")
    print(f"  Bayesian threshold:  {threshold_bayes:.3f}  "
          f"95% CI [{bayes_lower:.3f}, {bayes_upper:.3f}]")


# ===================================================================
# Save
# ===================================================================
results_df = pd.DataFrame(results)
print("\n", results_df)
results_df.to_csv("Data/threshold_estimates.csv", index=False)