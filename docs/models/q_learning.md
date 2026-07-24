# Q-learning

Q-learning represents the expected value associated with each deck and updates
the value of the selected deck after observing an outcome.

## Value update

A general Q-learning update can be written as:

$$
Q_{t+1}(a_t)
=
Q_t(a_t)
+
\alpha
\left[
u_t - Q_t(a_t)
\right]
$$

where:

- \(Q_t(a_t)\) is the estimated value of the selected deck before the update;
- \(u_t\) is the utility or outcome observed on trial \(t\);
- \(\alpha\) is the learning-rate parameter.

Replace or extend this equation if the implementation in this project uses a
different update rule.

## Choice rule

Document the exact choice rule used by the project here, including how deck
values are converted into choice probabilities.

## Parameters

Document the interpretation and permitted range of each Q-learning parameter
here.

## Optimizer starting points

For each subject, the implementation evaluates the negative log-likelihood on
a regular grid over the learning-rate and inverse-temperature bounds. It then
selects up to five distinct grid-local minima by default and runs one bounded
L-BFGS-B optimization from each selected point.

A grid point is a local minimum when its NLL is no greater than the NLL at its
immediate horizontal, vertical, and diagonal neighbors. Connected tied minima
are treated as one flat plateau and contribute only one starting point. This
prevents several adjacent points from the same promising region from consuming
all optimizer starts.

If the grid contains fewer than the requested number of distinct local-minimum
regions, only the available regions are used.

## Implementation

See the [Q-learning API](../api/models/q_learning.md) for the Python
implementation.
