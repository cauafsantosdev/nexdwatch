"""Shared persistence lifecycle states."""

import enum


class Status(str, enum.Enum):
    """Lifecycle state shared by film and pending-log ingestion."""

    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"
    FILTERED = "FILTERED"
