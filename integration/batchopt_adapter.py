"""The one function that touches OD. See docs/argus_localization_spec.md section 8.

Everything upstream of this function stays fixed across retriever and matcher
swaps. Only this adapter changes at integration time.
"""

from core.types import LocalizationResult


def to_batchopt_measurements(result: LocalizationResult):
    """TODO(team): map tie_points into batch-opt's measurement format.

    Confirm: field names, coordinate frame (lat/lon vs ECEF), and
    covariance/weight per correspondence.
    """
    raise NotImplementedError
