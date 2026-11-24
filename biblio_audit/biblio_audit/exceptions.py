"""Explicit domain exceptions for `biblio-audit` (zero silent fallbacks)."""

from __future__ import annotations


class BiblioAuditError(Exception):
    """Base exception for all `biblio-audit` errors."""


class FormatDetectionError(BiblioAuditError):
    """Raised when an input export format cannot be identified or parsed."""


class MissingFieldError(BiblioAuditError):
    """Raised when a required bibliographic column/field is absent for the chosen network."""


class InsufficientNetworkError(BiblioAuditError):
    """Raised when a constructed bibliometric network is too small or sparse to cluster."""
