from __future__ import annotations


def temporal_split(n: int, val_weeks: int) -> tuple[slice, slice]:
    if n <= val_weeks + 5:
        raise ValueError("Series too short for validation window")
    train_slice = slice(0, n - val_weeks)
    val_slice = slice(n - val_weeks, n)
    return train_slice, val_slice
