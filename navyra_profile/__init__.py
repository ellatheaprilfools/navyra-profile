"""navyra-profile: measure how semantically repetitive your LLM traffic is.

Open measurement tooling from Navyra Ltd. Contains no acceleration engine.
"""
__version__ = "0.1.0-dev"

from .fingerprint import Fingerprinter, hamming
from .analyzer import analyse, TrafficReport
