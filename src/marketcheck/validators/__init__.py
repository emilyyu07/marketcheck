"""Validation rules — pluggable checks for OHLCV data quality."""

import marketcheck.validators.financial  # noqa: F401
import marketcheck.validators.numerical  # noqa: F401

# Import rule modules so @register decorators execute at import time.
import marketcheck.validators.structural  # noqa: F401
import marketcheck.validators.temporal  # noqa: F401
from marketcheck.validators.base import REGISTRY, ValidationRule, register  # noqa: F401
