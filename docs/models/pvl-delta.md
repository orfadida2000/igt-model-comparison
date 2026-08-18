# PVL-Delta

## Parameters

The optimizer parameter vector is ordered as:

1. `learning_rate` \(\phi\);
2. `outcome_sensitivity` \(\alpha\);
3. `loss_aversion` \(\lambda\);
4. `response_consistency` \(c\).

Configured bounds are approximately:

| Parameter | Bounds |
|---|---|
| learning rate | \([10^{-6}, 1-10^{-6}]\) |
| outcome sensitivity | \([10^{-6}, 2]\) |
| loss aversion | \([10^{-6}, 10]\) |
| response consistency | \([0,5]\) |

The epsilon-adjusted learning-rate endpoints provide numerically closed optimizer bounds while approximating the intended open endpoints.

## Subjective utility

Net outcomes are first divided by 100. For scaled outcome \(x_t\):

\[
u_t =
\begin{cases}
x_t^\alpha, & x_t \ge 0 \\
-\lambda(-x_t)^\alpha, & x_t < 0
\end{cases}
\]

This allows diminishing/amplified sensitivity to outcome magnitude and asymmetric weighting of losses.

## Delta update

Let \(E_t(d)\) be the expectancy of deck \(d\). For the chosen deck \(a_t\):

\[
\delta_t = u_t - E_t(a_t)
\]

\[
E_{t+1}(a_t) = E_t(a_t) + \phi\,\delta_t
\]

Unchosen deck expectancies are unchanged.

## Choice consistency

The implementation maps response consistency to softmax sensitivity as:

\[
\theta = 3^c - 1
\]

and then uses:

\[
P(a_t=d) =
\frac{\exp(\theta E_t(d))}
{\sum_j \exp(\theta E_t(j))}
\]

## Initialization

PVL-Delta uses scrambled Sobol initialization over its four-dimensional parameter box. The default number of starts is 32, and the default fixed seed is 42.

The start count is constrained to a positive power of two by the main CLI, matching the balanced Sobol construction used by the initializer.

## Relationship to Q-learning

For the relevant mapping, objective-outcome Q-learning is represented within PVL-Delta when outcome sensitivity and loss aversion are both 1 and response consistency is chosen so that its transformed softmax sensitivity matches Q-learning's inverse temperature:

\[
c = \frac{\log(\beta + 1)}{\log 3}
\]

This relationship is used by the PVL-Delta correction workflow to create additional warm starts from fitted Q-learning solutions.
