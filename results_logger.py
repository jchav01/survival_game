"""Utility to persist survival game results."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Final

RESULTS_FILE: Final[Path] = Path("game_results.csv")


def log_result(score: int, survival_ticks: int, fps: int) -> None:
    """Append the run results to the CSV file.

    Parameters
    ----------
    score: int
        Total number of preys eaten during the run.
    survival_ticks: int
        Lifetime expressed in simulation ticks.
    fps: int
        Frames per second used for rendering, to convert ticks into seconds.
    """
    survival_seconds = survival_ticks / fps if fps else float(survival_ticks)
    header = "timestamp,score,survival_ticks,survival_seconds\n"
    line = (
        f"{datetime.now().isoformat(timespec='seconds')},"
        f"{score},"
        f"{survival_ticks},"
        f"{survival_seconds:.2f}\n"
    )

    file_exists = RESULTS_FILE.exists()
    with RESULTS_FILE.open("a", encoding="utf-8") as log_file:
        if not file_exists:
            log_file.write(header)
        log_file.write(line)
