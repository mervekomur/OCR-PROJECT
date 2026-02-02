"""
Logic module for FLO Masraf Modulu.
Contains business logic for tax calculation, semantic completion, etc.
"""

from .tax_calculator import TaxCalculator, SemanticCompleter

__all__ = ["TaxCalculator", "SemanticCompleter"]
