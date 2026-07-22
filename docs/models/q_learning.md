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

## Implementation

See the [Q-learning API](../api/models/q_learning.md) for the Python
implementation.
