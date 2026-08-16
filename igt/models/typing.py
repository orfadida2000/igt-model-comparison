from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from igt.constants.config import DECK_LABELS
from igt.typing import Float1DArray, Int1DArray


@dataclass(frozen=True, slots=True)
class SubjectData:
    """Trial-level Iowa Gambling Task data for one subject.

    Choices use the dataset's deck coding:

    - 1: deck A
    - 2: deck B
    - 3: deck C
    - 4: deck D

    Wins are expected to be non-negative and losses are expected to be
    non-positive.
    """

    choices: Int1DArray
    wins: Float1DArray
    losses: Float1DArray

    def __post_init__(self) -> None:
        """Validate the subject arrays."""

        if self.choices.ndim != 1:
            raise ValueError("choices must be a one-dimensional array.")

        if self.wins.ndim != 1:
            raise ValueError("wins must be a one-dimensional array.")

        if self.losses.ndim != 1:
            raise ValueError("losses must be a one-dimensional array.")

        n_trials = self.choices.size

        if n_trials == 0:
            raise ValueError("SubjectData must contain at least one trial.")

        if self.wins.size != n_trials or self.losses.size != n_trials:
            raise ValueError("choices, wins, and losses must contain the same number of trials.")

        if not np.isfinite(self.wins).all():
            raise ValueError("wins contains missing or non-finite values.")

        if not np.isfinite(self.losses).all():
            raise ValueError("losses contains missing or non-finite values.")

        valid_choices = np.isin(
            cast(ArrayLike, self.choices),
            np.array(list(DECK_LABELS.keys()), dtype=np.int64),
        )

        if not valid_choices.all():
            invalid_choices = np.unique(cast(ArrayLike, self.choices[~valid_choices])).tolist()

            raise ValueError(f"choices contains invalid deck codes: {invalid_choices}")

        if (self.wins < 0.0).any():
            raise ValueError("wins must contain only non-negative values.")

        if (self.losses > 0.0).any():
            raise ValueError("losses must contain only non-positive values.")

    @property
    def outcomes(self) -> Float1DArray:
        """Return the net outcome for every trial."""

        return self.wins + self.losses

    @property
    def n_trials(self) -> int:
        """Return the number of trials."""

        return int(self.choices.size)
