"""
File Summary:
- Exposes validation utilities for computing golden-number diagnostics.
- Provides convenience imports for sanity checks and report writing.
- Keeps validation logic decoupled from visualization modules.
"""

from .checks import compute_golden_numbers, assert_sane, write_validation_report

__all__ = ["compute_golden_numbers", "assert_sane", "write_validation_report"]
