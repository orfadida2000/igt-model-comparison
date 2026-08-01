# Q-learning

The project uses a one-state, four-action Q-learning model. Each action is one
IGT deck. The model maintains one learned value for each deck and updates only
the deck selected by the participant.

## Value update

For the selected deck \(a_t\):

$$
Q_{t+1}(a_t)
=
Q_t(a_t)
+
\alpha\left[r_t-Q_t(a_t)\right]
$$

where \(r_t\) is the net trial outcome after dividing the raw payoff by 100.
Unselected deck values remain unchanged.

The learning rate is bounded by:

$$
0 \leq \alpha \leq 1
$$

The endpoints remain available because they are meaningful model cases:
\(\alpha=0\) gives no learning and \(\alpha=1\) replaces the selected deck's
value with its latest outcome.

## Choice rule

Before each update, choice probabilities are calculated with a softmax rule:

$$
P_t(a=j)
=
\frac{\exp\left(\beta Q_t(j)\right)}
{\sum_k \exp\left(\beta Q_t(k)\right)}
$$

The inverse temperature \(\beta\) is bounded below by zero. Its upper bound is
a configurable optimization bound, with a default of 20.

## Optimizer starting points

For each subject, the implementation evaluates the NLL on a Cartesian grid and
selects up to five distinct grid-local minima by default. One bounded L-BFGS-B
optimization is run from each selected point.

The learning-rate dimension uses 31 quadratically spaced values by default:

$$
\alpha_i = u_i^2,
\qquad
u_i \in \operatorname{linspace}(0,1,31)
$$

This keeps both theoretical endpoints while adding substantially more grid
resolution below 0.05, where the initial full-data fits frequently placed the
continuous optimum.

The inverse-temperature dimension remains linearly spaced. When its grid size
is not supplied explicitly, the implementation chooses enough points to keep
approximately unit spacing. Therefore, upper bounds of 20, 50, and 100 produce
21, 51, and 101 inverse-temperature grid values respectively.

A grid point is a local minimum when its NLL is no greater than the NLL at its
immediate horizontal, vertical, and diagonal neighbors. Connected tied minima
are treated as one flat plateau and contribute only one starting point.

## Fit diagnostics

The output records the uniform-choice NLL, improvement over uniform choice,
whether the fitted likelihood is effectively uniform, and how many parameters
are at lower or upper bounds. Boundary estimates are diagnostics and do not by
themselves invalidate a fit.

## Implementation

See the [Q-learning API](../api/models/q_learning.md) for the Python
implementation.
