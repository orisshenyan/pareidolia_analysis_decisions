# ===================================================================
# Bayesian psychometric threshold estimation
# ===================================================================
#
#   For each participant, estimate the opacity level at which they
#   achieve 65% accuracy on a 2AFC face-direction task, using:
#
#     1. A Bayesian model with an informative prior (from pilot data)
#     2. A Bayesian model with a flat (uninformative) prior
#     3. A classical frequentist logistic regression (GLM)
#
#   The Bayesian models assume a fixed psychometric slope from pilot
#   data and estimate only the horizontal location (threshold) of the
#   psychometric curve. 
#   They use the PyMC package - PyMC:
#   A Modern and Comprehensive Probabilistic Programming Framework in Python,
#   Abril-Pla O, Andreani V, Carroll C, Dong L, Fonnesbeck CJ, Kochurov M, Kumar R,
#   Lao J, Luhmann CC, Martin OA, Osthege M, Vieira R, Wiecki T, Zinkov R. (2023)
#
#   The GLM freely estimates both intercept and slope per participant.
#
# Output:
#   A CSV file ("threshold_estimates.csv") containing three threshold
#   estimates per participant, plus Bayesian 95% HDI bounds.
# ===================================================================

# --- Libraries ---
import pandas as pd                    # tabular data handling
import glob                            # wildcard file/folder matching
import numpy as np                     # numerical operations
import pymc as pm                      # Bayesian probabilistic modelling
import arviz as az                     # Bayesian posterior diagnostics and summaries
import statsmodels.api as sm           # classical statistical models
import statsmodels.formula.api as smf  # formula-based model interface
import os                              # file and directory path handling


# ===================================================================
# Prior settings from pilot data
# ===================================================================
# prior_mean:
#   The best estimate of the psychometric midpoint (65% point on the
#   sigmoid) from earlier pilot work. This is the centre of the
#   informative normal prior placed on the threshold parameter.

# A note for Matteo: should we be using the bootstapped mean? 
#
# prior_sd:
#   The uncertainty (standard deviation) around the pilot estimate.
#   A wider SD makes the prior weaker and lets each participant's
#   data dominate the posterior more.
#
# slope:
#   The steepness of the psychometric function, held fixed across
#   all participants. 
#
# target:
#   The performance level of interest. We want the opacity value
#   at which a participant is predicted to be 65% correct.

# random_seed:
#   for reproducibility
# ===================================================================
prior_mean = 0.264
prior_sd   = 0.085
slope      = 7.60
target     = 0.65
randomseed = 31032026


# ===================================================================
# Import all subjects
# ===================================================================
# Each participant's training data is stored in a separate folder
# inside "Data/individual_data/".
# ===================================================================
subject_dirs = [
    d for d in glob.glob("Data/individual_data/*/")
    if os.path.isdir(d)
]
subject_dirs.sort()

results = []

# ===================================================================
# Main loop: process each participant
# ===================================================================
for subject_dir in subject_dirs:

    # Extract the subject identifier from the folder name.
    # e.g. "Data/individual_data/sub001/" → "sub001"
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

    # Concatenate all blocks into a single dataframe of all training
    # trials for this participant, and reset the row index.
    all_blocks = pd.concat([pd.read_csv(f) for f in block_files])
    all_blocks = all_blocks.reset_index(drop=True)

    # ---------------------------------------------------------------
    # 2. Score objective accuracy
    # ---------------------------------------------------------------
    # The 2AFC task asks participants which direction a face is
    # pointing: left (Direction_Report == 1) or right (== 2).
    #
    # Ground truth is encoded by Face_Position:
    #   negative = face on the left
    #   positive = face on the right
    #
    # A trial is correct (1) if report and position agree; else 0.
    all_blocks['Correct_Response'] = (
        ((all_blocks['Direction_Report'] == 1) & (all_blocks['Face_Position'] < 0)) |
        ((all_blocks['Direction_Report'] == 2) & (all_blocks['Face_Position'] > 0))
    ).astype(int)

    # ---------------------------------------------------------------
    # 3. Summarise accuracy at each opacity level
    # ---------------------------------------------------------------
    # Collapse trial-level data to binomial counts per opacity:
    #   k = number correct
    #   n = total trials
    opacity_levels = (
        all_blocks
        .groupby('Opacity')['Correct_Response']
        .agg(['sum', 'count'])
    )

    x = opacity_levels.index.values.astype(float)  # opacity values
    k = opacity_levels['sum'].values                # correct counts
    n = opacity_levels['count'].values              # trial counts

    # ---------------------------------------------------------------
    # 4a. Bayesian model — informative prior
    # ---------------------------------------------------------------
    # Model:
    #   p(correct | opacity) = sigmoid(slope × (opacity − threshold))
    #
    # Prior:
    #   threshold ~ Normal(prior_mean, prior_sd)
    #
    # The sigmoid midpoint corresponds to 50% accuracy. After
    # sampling, we convert each posterior draw to the 65% point
    # using the inverse logistic equation (see step 6).
    # ---------------------------------------------------------------
    with pm.Model() as model_informative:

        threshold = pm.Normal('threshold', mu=prior_mean, sigma=prior_sd)
        p_correct = pm.math.sigmoid(slope * (x - threshold))
        obs       = pm.Binomial('obs', n=n, p=p_correct, observed=k)

        trace_informative = pm.sample(
            2000,                      # posterior draws per chain (after tuning)
            chains=4,                  # independent MCMC chains
            target_accept=0.95,        # higher acceptance reduces divergences
            return_inferencedata=True,  # return an ArviZ InferenceData object
            progressbar=True,
            random_seed = randomseed
        )

    # ---------------------------------------------------------------
    # 4b. Bayesian model — flat (uninformative) prior
    # ---------------------------------------------------------------
    # Identical model structure, but the prior on threshold is now
    # Uniform(0.01, 0.99), expressing minimal prior knowledge.
    # ---------------------------------------------------------------
    with pm.Model() as model_flat:

        threshold_flat = pm.Uniform('threshold', lower=0.01, upper=0.99)
        p_correct_flat = pm.math.sigmoid(slope * (x - threshold_flat))
        obs_flat       = pm.Binomial('obs', n=n, p=p_correct_flat, observed=k)

        trace_flat = pm.sample(
            2000,
            chains=4,
            target_accept=0.95,
            return_inferencedata=True,
            progressbar=True,
            random_seed = randomseed
        )

    # ---------------------------------------------------------------
    # 5. Convert posterior threshold samples to 65%-correct opacity
    # ---------------------------------------------------------------
    # The threshold parameter in the sigmoid corresponds to the 50%
    # point. To find the opacity for any other target performance:
    #
    #   target = sigmoid(slope × (opacity − threshold))
    #
    # Solving for opacity:
    #
    #   logit(target) = slope × (opacity − threshold)
    #   opacity       = threshold + logit(target) / slope
    #
    # We apply this transform to every posterior sample so that the
    # full posterior distribution over the 65% threshold is preserved.
    # ---------------------------------------------------------------
    logit_target = np.log(target / (1 - target))

    # --- Informative prior model ---
    samples_informative    = trace_informative.posterior['threshold'].values.flatten()
    samples_informative_65 = samples_informative + (logit_target / slope)
    posterior_mean_info    = samples_informative_65.mean()
    hdi_info               = az.hdi(samples_informative_65, hdi_prob=0.95)

    # --- Flat prior model ---
    samples_flat    = trace_flat.posterior['threshold'].values.flatten()
    samples_flat_65 = samples_flat + (logit_target / slope)
    posterior_mean_flat = samples_flat_65.mean()
    hdi_flat            = az.hdi(samples_flat_65, hdi_prob=0.95)

    # ---------------------------------------------------------------
    # 6. Classical logistic regression (GLM)
    # ---------------------------------------------------------------
    # Fit a standard frequentist logistic regression at the trial level:
    #
    #   logit(p(correct)) = β₀ + β₁ × Opacity
    #
    # Unlike the Bayesian models, this freely estimates BOTH intercept
    # and slope from the participant's data alone (no prior, no fixed
    # slope). With few trials, this can produce extreme or unreliable
    # estimates — which is one motivation for the Bayesian approach.
    #
    # To find the 65% threshold, solve:
    #
    #   logit(0.65) = β₀ + β₁ × opacity
    #   opacity     = (logit(0.65) − β₀) / β₁
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
    # 7. Store results for this participant
    # ---------------------------------------------------------------
    results.append({
        'subject':                subject_id,

        # Frequentist estimate
        'glm_threshold':          glm_threshold_65,

        # Bayesian estimate — informative prior
        'bayesian_informative':   posterior_mean_info,
        'bayesian_info_hdi_low':  hdi_info[0],
        'bayesian_info_hdi_high': hdi_info[1],

        # Bayesian estimate — flat prior
        'bayesian_flat':          posterior_mean_flat,
        'bayesian_flat_hdi_low':  hdi_flat[0],
        'bayesian_flat_hdi_high': hdi_flat[1],
    })

    print(f"  GLM threshold:              {glm_threshold_65:.3f}")
    print(f"  Bayesian (informative):     {posterior_mean_info:.3f}  "
          f"95% HDI [{hdi_info[0]:.3f}, {hdi_info[1]:.3f}]")
    print(f"  Bayesian (flat prior):      {posterior_mean_flat:.3f}  "
          f"95% HDI [{hdi_flat[0]:.3f}, {hdi_flat[1]:.3f}]")


# ===================================================================
# Build final results dataframe and save
# ===================================================================
results_df = pd.DataFrame(results)

# Display to console
print("\n", results_df)

# Save to CSV
results_df.to_csv("Data/threshold_estimates.csv", index=False)