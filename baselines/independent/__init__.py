"""Independent source-calibration baseline."""

from typing import Any, Sequence

import numpy as np

from .._common import measured_accuracy


class Independent:
    """Estimate every model separately from its labeled source accuracy."""

    name = "Independent"

    def evaluate(
        self, source_responses: Sequence[Sequence[Any]], source_gold: Sequence[Any]
    ) -> np.ndarray:
        return measured_accuracy(source_responses, source_gold)
