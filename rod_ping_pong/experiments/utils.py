"""Utilities for experimentation"""

import time
from typing import Any, Self
import numpy as np

class Stopwatch:
    """Uses context manager to measure time for a block of code."""

    def __init__(self) -> None:
        """Initialize the class."""
        self.start_time = 0.0
        self.total_iterations = 0
        self.total_elapsed = 0.0
        self.elapsed = 0.0
        self.cumulative_elapsed = 0.0
        self.cumulative_iterations = 0

    def __enter__(self) -> Self:
        """Start the stopwatch.

        Returns:
            self
        """
        self.start_time = time.time()
        return self

    def __exit__(self, *args: Any) -> None:  # noqa D105
        """Stop the stopwatch and save the elapsed time."""
        self.elapsed = time.time() - self.start_time
        self.total_elapsed += self.elapsed
        self.cumulative_elapsed += self.elapsed
        self.cumulative_iterations += 1
        self.total_iterations += 1

    @property
    def fps(self) -> float:
        """Return frames per second of last iteration."""
        if self.elapsed == 0:
            return np.nan
        return 1 / (self.elapsed)

    def get_avg_cum_time(self, reset_cum_elapsed: bool = False) -> float:
        """Return average time elapsed in seconds for iterations that were called after the last cumulative reset.

        Args:
            reset_cum_elapsed: whether to reset cumulative elapsed time in the stopwatch.

        Returns:
            average tracked time since last reset.
        """
        res = 0.0 if self.cumulative_iterations == 0 else self.cumulative_elapsed / self.cumulative_iterations

        if reset_cum_elapsed:
            self.reset_cum_elapsed()
        return res

    def get_avg_cum_fps(self, reset_cum_elapsed: bool = False) -> float:
        """Return average frames per second since last reset.

        Args:
            reset_cum_elapsed: boolean denoting whether to reset the cumulative time in the stopwatch.

        Returns:
            frames per second since last reset.
        """
        avg_time = self.get_avg_cum_time(reset_cum_elapsed)
        if avg_time == 0:
            return np.nan
        return 1 / avg_time

    def reset_cum_elapsed(self) -> None:
        """Reset cumulative time in the stopwatch."""
        self.cumulative_elapsed = 0
        self.cumulative_iterations = 0

