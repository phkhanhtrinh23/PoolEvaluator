"""Agreement-on-the-Line accuracy estimator."""

from typing import Any, Sequence

import numpy as np

from .._common import mean_pairwise_agreement, measured_accuracy


class AgreementLine:
    """Fit the source agreement--accuracy line and apply it to target agreement."""

    name = "Agreement-on-the-Line"

    def evaluate(
        self,
        target_responses: Sequence[Sequence[Any]],
        source_responses: Sequence[Sequence[Any]],
        source_gold: Sequence[Any],
    ) -> np.ndarray:
        source_agreement = mean_pairwise_agreement(source_responses)
        target_agreement = mean_pairwise_agreement(target_responses)
        source_accuracy = measured_accuracy(source_responses, source_gold)
        design = np.column_stack([source_agreement, np.ones_like(source_agreement)])
        slope, intercept = np.linalg.lstsq(design, source_accuracy, rcond=None)[0]
        return np.clip(slope * target_agreement + intercept, 0.0, 1.0)
