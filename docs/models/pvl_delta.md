# PVL-Delta

PVL-Delta combines a subjective utility function, a delta learning rule, and a
probabilistic choice rule.

## Utility function

Document the exact utility function used by the implementation here.

Include:

- the treatment of gains;
- the treatment of losses;
- the outcome-sensitivity parameter;
- the loss-aversion parameter.

## Learning rule

After deck \(a_t\) is selected on trial \(t\), its expectancy is updated using
the delta rule:

$$
E_{t+1}(a_t)
=
E_t(a_t)
+
\phi
\left[
u_t - E_t(a_t)
\right]
$$

where:

- \(E_t(a_t)\) is the expectancy of the selected deck before the update;
- \(u_t\) is the subjective utility of the outcome observed on trial \(t\);
- \(\phi\) is the recency or learning-rate parameter.

The expectancies of the unselected decks remain unchanged:

$$
E_{t+1}(a)
=
E_t(a),
\qquad a \ne a_t
$$

## Choice rule

Document how the model converts deck expectancies into choice probabilities.

## Parameters

Document the interpretation and permitted range of each PVL-Delta parameter
here.

## Implementation

See the [PVL-Delta API](../api/models/pvl_delta.md) for the Python
implementation.
