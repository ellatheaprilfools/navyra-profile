"""
navyra_profile.validation — accuracy validation harness.

PROJECT TASK 5. Runs synthetic traffic with known ground truth through
analyse(), and compares reported tier rates against the known truth,
across a range of template_share values — quantifying the profiler's
per-tier accuracy rather than just asserting it.
"""

from __future__ import annotations
