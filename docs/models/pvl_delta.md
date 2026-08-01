# PVL-Delta

PVL-Delta combines a subjective utility function, a delta learning rule, and a
probabilistic choice rule.

## Utility function

For the scaled net outcome \(x_t\):

$$
u(x_t)=
\begin{cases}
{x_t}^{\alpha}, & x_t \geq 0 \\
-\lambda\lvert x_t\rvert^{\alpha}, & x_t < 0
\end{cases}
$$

where \(\alpha\) is outcome sensitivity and \(\lambda\) is loss aversion.
Raw IGT payoffs are divided by 100 before this transformation.

## Learning rule

After deck \(a_t\) is selected, its expectancy is updated using:

$$
E_{t+1}(a_t)
=
E_t(a_t)
+
A\left[u(x_t)-E_t(a_t)\right]
$$

The expectancies of unselected decks remain unchanged.

## Choice rule

Response consistency \(c\) is transformed into softmax sensitivity:

$$
\theta=3^c-1
$$

and the deck probabilities are:

$$
P_t(a=j)
=
\frac{\exp\left(\theta E_t(j)\right)}
{\sum_k \exp\left(\theta E_t(k)\right)}
$$

## Parameters and bounds

The implementation uses the expanded PVL-Delta parameterization:

- learning rate \(A\): approximately \((0,1)\), represented with epsilon bounds;
- outcome sensitivity \(\alpha\): approximately \((0,2]\);
- loss aversion \(\lambda\): approximately \((0,10]\);
- response consistency \(c\): \([0,5]\).

## Optimizer starting points

A scrambled Sobol sequence is generated once when the model object is created
and the same stored starting-point array is reused for every subject. The
default is 32 starts. The main workflow uses a fixed seed, making the Sobol
points reproducible across complete executions.

## Fit diagnostics

The output records the uniform-choice NLL, improvement over uniform choice,
whether the fitted likelihood is effectively uniform, and how many parameters
are at lower or upper bounds. Boundary estimates are diagnostics and do not by
themselves invalidate a fit.

## Implementation

See the [PVL-Delta API](../api/models/pvl_delta.md) for the Python
implementation.
