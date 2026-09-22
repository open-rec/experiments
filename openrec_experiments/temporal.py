"""Timestamp conversion shared by experiment protocols."""
import pandas as pd


def timestamp(value):
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError("split boundaries must include a UTC offset")
    return result.value // 1_000_000


def utc_milliseconds(values):
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    if parsed.isna().any():
        raise ValueError("missing event timestamp")
    return parsed.astype("datetime64[ns, UTC]").astype("int64") // 1_000_000
