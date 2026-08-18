"""Validated subject-level arrays consumed by computational models.

`SubjectData` stores choices, gains, and losses for one participant, enforces shape
and value invariants, and exposes total outcomes and trial count used by model
likelihood functions.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from igt.constants.config import DECK_LABELS
from igt.typing import Float1DArray, Int1DArray


@dataclass(frozen=True, slots=True)
class SubjectData:
    """Validated subject-level choices and monetary outcomes for model fitting.

    All arrays must be one-dimensional, have equal length, contain finite numeric
    values, and encode deck choices as integers from 1 through 4. The immutable
    container is the common input representation for
    [`ComputationalModel`][igt.models.base.ComputationalModel] likelihood methods.

    Attributes:
        choices: Chosen deck number for each trial.
        wins: Monetary win observed on each trial.
        losses: Monetary loss observed on each trial.
    """

    choices: Int1DArray
    wins: Float1DArray
    losses: Float1DArray

    def __post_init__(self) -> None:
        """Validate and normalize the subject-level model input arrays.

        The three arrays must be one-dimensional, equal in length, nonempty, and finite.
        Choices must be integer-valued deck identifiers from 1 through 4. Validated
        arrays are stored as independent NumPy arrays with the project's canonical
        dtypes.

        Raises:
            ValueError: If the arrays have incompatible shapes or lengths, are empty,
                contain non-finite values, or contain an invalid deck choice.
        """

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
        """Return net monetary outcome on each trial.

        Returns:
            Elementwise wins plus losses for the participant.
        """

        return self.wins + self.losses

    @property
    def n_trials(self) -> int:
        """Return the number of trials represented by the subject data.

        Returns:
            Number of choice observations.
        """

        return int(self.choices.size)
