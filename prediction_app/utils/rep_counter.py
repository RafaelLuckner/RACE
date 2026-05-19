"""Repetition counting via state machine with hysteresis.

Same logic as notebook 5 (5-exercise_evaluation_counting.ipynb), adapted for
frame-by-frame streaming inside the prediction pipeline.

State transitions:
  UNKNOWN/EXTENDED  → CONTRACTED  when angle < contracted_thresh
  CONTRACTED        → EXTENDED    when angle > extended_thresh  (+1 rep)

The hysteresis gap (contracted_thresh < extended_thresh) prevents false
triggers caused by noise in the angular signal.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict


# ── Exercise configuration ────────────────────────────────────────────────────
# joint        : which angle to watch (maps to 'right_{joint}' / 'left_{joint}')
# extended_thresh  : angle above which the joint is considered "extended"
# contracted_thresh: angle below which the joint is considered "contracted"
EXERCISE_REP_CONFIG: Dict[str, Dict] = {
    "flexao": {
        "joint": "cotovelo",
        "extended_thresh": 130,
        "contracted_thresh": 90,
    },
    "agachamento": {
        "joint": "joelho",
        "extended_thresh": 140,
        "contracted_thresh": 100,
    },
    "rosca_biceps": {
        "joint": "cotovelo",
        "extended_thresh": 140,
        "contracted_thresh": 80,
    },
}


class _RepState(Enum):
    UNKNOWN = 0
    EXTENDED = 1
    CONTRACTED = 2


class RepetitionCounter:
    """
    Stateful per-exercise repetition counter.

    Feed each frame's bilateral angle with ``update(exercise, angle)``.
    A rep is registered on the CONTRACTED → EXTENDED transition.
    The counter accumulates across calls — call ``reset()`` to start fresh.
    """

    def __init__(self) -> None:
        self._states: Dict[str, _RepState] = {
            ex: _RepState.UNKNOWN for ex in EXERCISE_REP_CONFIG
        }
        self._counts: Dict[str, int] = {ex: 0 for ex in EXERCISE_REP_CONFIG}

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all states and counts to zero."""
        self._states = {ex: _RepState.UNKNOWN for ex in EXERCISE_REP_CONFIG}
        self._counts = {ex: 0 for ex in EXERCISE_REP_CONFIG}

    # ------------------------------------------------------------------
    def update(self, exercise: str, bilateral_angle: float) -> bool:
        """
        Feed one frame's bilateral weighted angle for *exercise*.

        Parameters
        ----------
        exercise:
            Exercise name, e.g. ``"flexao"``, ``"agachamento"``, ``"rosca_biceps"``.
        bilateral_angle:
            Angle in degrees (bilateral visibility-weighted).

        Returns
        -------
        bool
            ``True`` if a complete repetition was registered on this frame.
        """
        if exercise not in EXERCISE_REP_CONFIG:
            return False

        cfg = EXERCISE_REP_CONFIG[exercise]
        ext_t = cfg["extended_thresh"]
        cnt_t = cfg["contracted_thresh"]
        state = self._states[exercise]
        completed = False

        if state == _RepState.UNKNOWN:
            if bilateral_angle > ext_t:
                self._states[exercise] = _RepState.EXTENDED
            elif bilateral_angle < cnt_t:
                self._states[exercise] = _RepState.CONTRACTED

        elif state == _RepState.EXTENDED:
            if bilateral_angle < cnt_t:
                self._states[exercise] = _RepState.CONTRACTED

        elif state == _RepState.CONTRACTED:
            if bilateral_angle > ext_t:
                self._states[exercise] = _RepState.EXTENDED
                self._counts[exercise] += 1
                completed = True

        return completed

    # ------------------------------------------------------------------
    @property
    def counts(self) -> Dict[str, int]:
        """Current rep counts per exercise (copy)."""
        return dict(self._counts)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"RepetitionCounter({self._counts})"
