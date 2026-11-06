"""Unit tests for SortStateMachine using synthetic feature sequences.

Run with:
    python -m pytest tests/test_state_machine.py -v
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from live import (  # noqa: E402
    FrameFeatures,
    SortStateMachine,
    State,
    _SMConfig,
    update_empty_ref,
)
from common import get_classes  # noqa: E402

_CLASSES = get_classes()


# ─────────────────────────────────────────────────────────────────────────────
# Shared config & dummy data
# ─────────────────────────────────────────────────────────────────────────────

STABLE_FRAMES = 3
VOTE_FRAMES   = 4

_BASE_CFG = _SMConfig(
    empty_diff_threshold  = 0.02,
    stable_diff_threshold = 0.005,
    stable_frames         = STABLE_FRAMES,
    vote_frames           = VOTE_FRAMES,
    confidence_threshold  = 0.70,
    release_threshold     = 0.03,
    clear_timeout_s       = 20.0,
    empty_label           = _CLASSES[-1],
    fallback_label        = _CLASSES[-2],
)

_DUMMY_ROI = np.zeros((64, 64, 3), dtype=np.uint8)

_EMPTY_FEAT    = FrameFeatures(diff_to_empty=0.00, diff_to_prev=0.00)   # nothing on tray
_PRESENT_FEAT  = FrameFeatures(diff_to_empty=0.05, diff_to_prev=0.05)   # object moving
_SETTLED_FEAT  = FrameFeatures(diff_to_empty=0.05, diff_to_prev=0.001)  # object still
_CLEARING_FEAT = FrameFeatures(diff_to_empty=0.00, diff_to_prev=0.001)  # tray clearing


def _make_sm(
    label: str = _CLASSES[0],
    confidence: float = 0.90,
    on_decision=None,
    on_false_trigger=None,
    cfg: _SMConfig | None = None,
) -> SortStateMachine:
    classify_fn = MagicMock(return_value=(label, confidence))
    return SortStateMachine(
        cfg=replace(_BASE_CFG) if cfg is None else cfg,
        classify_fn=classify_fn,
        on_decision=on_decision,
        on_false_trigger=on_false_trigger,
    )


def _drive(sm: SortStateMachine, features: FrameFeatures, n: int) -> list[dict | None]:
    return [sm.tick(features, _DUMMY_ROI) for _ in range(n)]


def _advance_to_classify(sm: SortStateMachine) -> None:
    """Drive sm from IDLE to CLASSIFY (the first vote-collection tick).

    Sequence:
      1 tick  PRESENT_FEAT  → PRESENT
      N ticks SETTLED_FEAT  → STABLE  (after stable_frames)
      1 tick  SETTLED_FEAT  → CLASSIFY (STABLE is consumed in one tick)
    """
    sm.tick(_PRESENT_FEAT, _DUMMY_ROI)
    assert sm.state == State.PRESENT
    _drive(sm, _SETTLED_FEAT, STABLE_FRAMES)
    assert sm.state == State.STABLE
    sm.tick(_SETTLED_FEAT, _DUMMY_ROI)   # consume STABLE → CLASSIFY
    assert sm.state == State.CLASSIFY


def _full_cycle(sm: SortStateMachine) -> dict:
    """Drive sm from IDLE through exactly one decision and return the event."""
    _advance_to_classify(sm)

    # collect votes
    _drive(sm, _SETTLED_FEAT, VOTE_FRAMES)
    assert sm.state == State.DECIDE

    # fire decision
    event = sm.tick(_CLEARING_FEAT, _DUMMY_ROI)
    assert sm.state == State.WAIT_CLEAR
    assert event is not None
    return event


# ─────────────────────────────────────────────────────────────────────────────
# State-transition tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIdleTransitions:
    def test_stays_idle_when_empty(self):
        sm = _make_sm()
        _drive(sm, _EMPTY_FEAT, 10)
        assert sm.state == State.IDLE

    def test_moves_to_present_on_diff(self):
        sm = _make_sm()
        sm.tick(_PRESENT_FEAT, _DUMMY_ROI)
        assert sm.state == State.PRESENT


class TestPresentTransitions:
    def test_back_to_idle_if_diff_drops(self):
        sm = _make_sm()
        sm.tick(_PRESENT_FEAT, _DUMMY_ROI)   # → PRESENT
        sm.tick(_EMPTY_FEAT,   _DUMMY_ROI)   # object gone
        assert sm.state == State.IDLE

    def test_resets_counter_on_motion(self):
        sm = _make_sm()
        sm.tick(_PRESENT_FEAT, _DUMMY_ROI)   # → PRESENT
        sm.tick(_SETTLED_FEAT, _DUMMY_ROI)   # consec=1
        sm.tick(_SETTLED_FEAT, _DUMMY_ROI)   # consec=2
        sm.tick(_PRESENT_FEAT, _DUMMY_ROI)   # jitter → consec=0
        assert sm.state == State.PRESENT
        assert sm._consec == 0

    def test_reaches_stable_after_enough_quiet_frames(self):
        sm = _make_sm()
        sm.tick(_PRESENT_FEAT, _DUMMY_ROI)
        _drive(sm, _SETTLED_FEAT, STABLE_FRAMES)
        assert sm.state == State.STABLE     # STABLE requires one more tick to move on


class TestStableTransition:
    def test_stable_advances_to_classify_on_next_tick(self):
        sm = _make_sm()
        sm.tick(_PRESENT_FEAT, _DUMMY_ROI)
        _drive(sm, _SETTLED_FEAT, STABLE_FRAMES)
        assert sm.state == State.STABLE
        sm.tick(_SETTLED_FEAT, _DUMMY_ROI)
        assert sm.state == State.CLASSIFY


class TestClassifyAndDecide:
    def test_collects_correct_vote_count(self):
        sm = _make_sm(label=_CLASSES[1], confidence=0.85)
        _advance_to_classify(sm)
        assert sm.state == State.CLASSIFY
        _drive(sm, _SETTLED_FEAT, VOTE_FRAMES)
        assert sm.state == State.DECIDE
        assert len(sm._votes) == VOTE_FRAMES

    def test_majority_label_wins(self):
        # Three votes for one class and one for another; the majority wins.
        labels   = [_CLASSES[2], _CLASSES[2], _CLASSES[2], _CLASSES[3]]
        confs    = [0.90,    0.88,    0.91,    0.80]
        call_seq = iter(zip(labels, confs))
        sm = SortStateMachine(
            cfg=_BASE_CFG,
            classify_fn=lambda _roi: next(call_seq),
        )
        event = _full_cycle(sm)
        assert event["label"] == _CLASSES[2]

    def test_confident_event_not_uncertain(self):
        sm = _make_sm(label=_CLASSES[3], confidence=0.95)
        event = _full_cycle(sm)
        assert event["uncertain"] is False
        assert event["label"] == _CLASSES[3]

    def test_low_confidence_becomes_other(self):
        sm = _make_sm(label=_CLASSES[0], confidence=0.50)
        event = _full_cycle(sm)
        assert event["uncertain"] is True
        assert event["label"] == _CLASSES[-2]

    def test_event_has_required_keys(self):
        sm = _make_sm()
        event = _full_cycle(sm)
        for key in ("ts", "label", "confidence", "uncertain", "thumb"):
            assert key in event

    def test_thumb_is_data_uri(self):
        sm = _make_sm()
        event = _full_cycle(sm)
        assert event["thumb"].startswith("data:image/jpeg;base64,")

    def test_on_decision_callback_called_once(self):
        cb = MagicMock()
        sm = _make_sm(on_decision=cb)
        _full_cycle(sm)
        cb.assert_called_once()
        assert cb.call_args[0][0]["label"] in (_CLASSES[0], _CLASSES[-2])


class TestWaitClear:
    def test_stays_in_wait_clear_while_object_present(self):
        sm = _make_sm()
        _full_cycle(sm)
        assert sm.state == State.WAIT_CLEAR
        _drive(sm, _PRESENT_FEAT, 10)
        assert sm.state == State.WAIT_CLEAR

    def test_returns_to_idle_after_clear(self):
        sm = _make_sm()
        _full_cycle(sm)
        _drive(sm, _CLEARING_FEAT, STABLE_FRAMES)
        assert sm.state == State.IDLE

    def test_counter_resets_on_object_return(self):
        sm = _make_sm()
        _full_cycle(sm)
        # partially clear, then object comes back
        _drive(sm, _CLEARING_FEAT, STABLE_FRAMES - 1)
        sm.tick(_PRESENT_FEAT, _DUMMY_ROI)   # object back → counter reset
        assert sm.state == State.WAIT_CLEAR
        assert sm._consec == 0

    def test_loose_release_threshold_clears_lighting_drift(self):
        sm = _make_sm()
        _full_cycle(sm)
        drift = FrameFeatures(
            diff_to_empty=_BASE_CFG.empty_diff_threshold * 1.3,
            diff_to_prev=0.001,
        )
        _drive(sm, drift, STABLE_FRAMES)
        assert sm.state == State.IDLE

    def test_model_assisted_release_after_consecutive_empty_checks(self):
        empty_label = get_classes()[-1]
        classify_calls = 0

        def classify(_roi):
            nonlocal classify_calls
            classify_calls += 1
            if classify_calls <= VOTE_FRAMES:
                return _CLASSES[0], 0.9
            return empty_label, 0.9

        sm = SortStateMachine(
            cfg=replace(_BASE_CFG),
            classify_fn=classify,
        )
        _advance_to_classify(sm)
        _drive(sm, _SETTLED_FEAT, VOTE_FRAMES)
        sm.tick(_PRESENT_FEAT, _DUMMY_ROI)
        assert sm.state == State.WAIT_CLEAR
        _drive(sm, _PRESENT_FEAT, VOTE_FRAMES * 5)
        assert sm.state == State.IDLE
        assert classify_calls == VOTE_FRAMES * 2

    def test_timeout_message_then_continue_recaptures_reference(self):
        sm = _make_sm()
        _full_cycle(sm)
        started = sm._clear_started_at
        assert started is not None
        sm.tick(_PRESENT_FEAT, _DUMMY_ROI, now=started + _BASE_CFG.clear_timeout_s + 1)
        assert sm.state == State.WAIT_CLEAR
        assert sm.message == "Remove the item, then press Continue"
        sm.request_continue()
        sm.tick(_PRESENT_FEAT, _DUMMY_ROI)
        assert sm.state == State.IDLE
        assert sm.consume_reference_recapture() is True
        assert sm.message == ""

    def test_restart_from_every_state(self):
        for state in State:
            sm = _make_sm()
            sm.state = state
            sm._votes = [(_CLASSES[0], 0.9)]
            sm.request_restart()
            sm.tick(_PRESENT_FEAT, _DUMMY_ROI)
            assert sm.state == State.IDLE
            assert sm._votes == []
            assert sm.consume_reference_recapture() is True

    def test_empty_classification_is_logged_as_false_trigger_not_event(self):
        empty_label = get_classes()[-1]
        event_callback = MagicMock()
        false_callback = MagicMock()
        sm = _make_sm(
            label=empty_label,
            on_decision=event_callback,
            on_false_trigger=false_callback,
        )
        _advance_to_classify(sm)
        _drive(sm, _SETTLED_FEAT, VOTE_FRAMES)
        event = sm.tick(_SETTLED_FEAT, _DUMMY_ROI)
        assert event is None
        assert sm.state == State.IDLE
        assert sm.false_trigger_count == 1
        event_callback.assert_not_called()
        false_callback.assert_called_once()

    def test_empty_reference_ema_tracks_calm_lighting(self):
        old_reference = np.zeros((8, 8), dtype=np.uint8)
        current_gray = np.full((8, 8), 100, dtype=np.uint8)
        updated = update_empty_ref(old_reference, current_gray, 0.02)
        assert np.all(updated == 2)


class TestReset:
    def test_reset_returns_to_idle(self):
        sm = _make_sm()
        sm.tick(_PRESENT_FEAT, _DUMMY_ROI)
        sm.reset()
        assert sm.state == State.IDLE
        assert sm._consec == 0
        assert sm._votes == []


# ─────────────────────────────────────────────────────────────────────────────
# Full-cycle integration: exactly one event per item
# ─────────────────────────────────────────────────────────────────────────────

class TestFullCycleIntegration:
    def _run_items(self, n_items: int) -> list[dict]:
        """Simulate n_items objects placed and removed; collect events via callback."""
        events: list[dict] = []
        sm = _make_sm(label=_CLASSES[2], confidence=0.88,
                      on_decision=lambda e: events.append(e))

        for _ in range(n_items):
            _advance_to_classify(sm)
            _drive(sm, _SETTLED_FEAT, VOTE_FRAMES)    # → DECIDE
            sm.tick(_CLEARING_FEAT, _DUMMY_ROI)       # → WAIT_CLEAR, fires event
            _drive(sm, _CLEARING_FEAT, STABLE_FRAMES) # → IDLE

        return events

    def test_one_event_per_item(self):
        for n in (1, 2, 3):
            events = self._run_items(n)
            assert len(events) == n, f"Expected {n} events, got {len(events)}"

    def test_no_spurious_events_on_empty_tray(self):
        events: list[dict] = []
        sm = _make_sm(on_decision=lambda e: events.append(e))
        _drive(sm, _EMPTY_FEAT, 100)
        assert events == []
