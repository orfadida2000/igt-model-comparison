# Statistical inference

The project uses information criteria for per-subject model comparison and adds population-level inference across the paired subject results.

## Paired criterion differences

For subject \(i\):

\[
\Delta AIC_i = AIC_{Q,i} - AIC_{PVL,i}
\]

\[
\Delta BIC_i = BIC_{Q,i} - BIC_{PVL,i}
\]

A positive difference favors PVL-Delta because lower information-criterion values are better.

## BCa bootstrap intervals

For both AIC and BIC differences, the analysis reports:

- sample mean;
- sample median;
- bias-corrected and accelerated (BCa) bootstrap confidence interval for the mean;
- BCa bootstrap confidence interval for the median.

The default analysis uses 10,000 resamples, 95% confidence, and seed 42.

Subjects are resampled as the observational units.

## Wilcoxon signed-rank tests

The analysis additionally performs a two-sided Wilcoxon signed-rank test on each paired criterion-difference vector.

This tests whether the paired differences show a systematic signed shift away from zero under the signed-rank assumptions. It should not be described simply as a test of the arithmetic mean.

The implementation uses SciPy's `wilcox` zero method and reports the test statistic, p-value, number of nonzero differences, and alternative.

## Model-win inference

For each criterion, every eligible subject is classified as:

- PVL-Delta win;
- Q-learning win;
- exact tie.

Exact ties are excluded from the win-rate denominator.

The analysis reports:

- PVL-Delta win proportion among non-tied subjects;
- exact binomial confidence interval;
- two-sided exact binomial test against a null PVL-Delta win probability of 0.5.

## Interpretation boundary

The inferential unit is the **subject**. The current analysis does not fit a hierarchical or cluster-robust model for the 10 source studies. Study-specific preference summaries are therefore descriptive indicators of heterogeneity rather than a replacement for a multilevel analysis.

AIC/BIC comparisons and the population-level tests should be interpreted as evidence about relative likelihood-based, complexity-penalized fit in the sampled participants, not as proof that either model is the true generative process.
