# Models

The project compares two reinforcement-learning-style models of IGT choice behavior.

| Model | Parameters | Outcome representation | Choice rule |
|---|---:|---|---|
| Q-learning | 2 | Scaled objective net outcome | Softmax with inverse temperature |
| PVL-Delta | 4 | Prospect-style subjective utility | Softmax with transformed response consistency |

Both models:

- maintain one learned value/expectancy for each of the four decks;
- update only the chosen deck;
- use a delta/prediction-error update;
- evaluate the observed choices through a trial-wise negative log-likelihood;
- are fitted independently for each participant.

See [Q-learning](q-learning.md) and [PVL-Delta](pvl-delta.md) for the exact implementation.
