"""Teleop DDS signal contract: plain-list encoding and the relayed toggle rule (no SDK)."""

from dataclasses import replace

import pytest

from vr_link.messages import teleop_dds as td
from vr_link.types import ControllerInput, VRInputState


def _controller(count=3, session=1, connected=True):
    return td.ControllerSample(
        record_toggle_count=count,
        session_id=session,
        connected=connected,
        left=(0.1, 0.2, 1.0, 0.0, 0.0, -0.5, 0.25),
        right=(0.9, 0.8, 0.0, 1.0, 1.0, 0.5, -0.25),
        stamp_ns=1_700_000_000_123_456_789,
    )


def test_controller_round_trip_and_layout():
    sample = _controller()
    frame_id, axes, buttons = td.encode_controller(sample)
    assert frame_id == td.CONTROLLER_FRAME_ID == "vr_link.controller.v1"
    assert len(axes) == td.CONTROLLER_AXIS_COUNT == 14 and len(buttons) == td.CONTROLLER_BUTTON_COUNT == 3
    assert buttons == [3, 1, 1]
    assert axes[td.CONTROLLER_LEFT_AXIS_OFFSET:td.CONTROLLER_RIGHT_AXIS_OFFSET] == list(sample.left)
    assert axes[td.CONTROLLER_RIGHT_AXIS_OFFSET:] == list(sample.right)
    assert td.decode_controller(frame_id, axes, buttons, stamp_ns=sample.stamp_ns) == sample
    # Appended fields from a newer sender are tolerated; shorter vectors are a contract violation.
    assert td.decode_controller(frame_id, axes + [9.0], buttons + [7], stamp_ns=sample.stamp_ns) == sample
    with pytest.raises(ValueError):
        td.decode_controller(frame_id, axes[:-1], buttons)
    with pytest.raises(ValueError):
        td.decode_controller(frame_id, axes, buttons[:2])


def test_controller_from_vr_state_follows_controller_input_order():
    state = VRInputState(
        left_input=ControllerInput(trigger=0.1, grip=0.2, button_first=1.0, button_second=0.0, button_thumbstick=0.0, thumbstick=(-0.5, 0.25)),
        right_input=ControllerInput(trigger=0.9, grip=0.8, button_first=0.0, button_second=1.0, button_thumbstick=1.0, thumbstick=(0.5, -0.25)),
        record_toggle_count=3,
    )
    sample = td.controller_sample_from_vr_state(state, session_id=1, connected=True, stamp_ns=5)
    assert sample == replace(_controller(), stamp_ns=5)
    assert td.CONTROLLER_AXIS_FIELDS == ("trigger", "grip", "button_first", "button_second", "button_thumbstick", "thumbstick_x", "thumbstick_y")


def test_foreign_frame_id_is_ignored_not_misread():
    _, axes, buttons = td.encode_controller(_controller())
    assert td.decode_controller("", axes, buttons) is None
    assert td.decode_controller("gamepad", axes, buttons) is None
    assert td.decode_recording_state(td.CONTROLLER_FRAME_ID, axes, buttons) is None
    assert td.decode_controller(td.RECORDING_STATE_FRAME_ID, [], [1, 0]) is None


def test_controller_encoding_rejects_out_of_contract_values():
    with pytest.raises(ValueError):
        td.encode_controller(_controller(count=-1))
    with pytest.raises(ValueError):
        td.encode_controller(_controller(count=2**31))
    with pytest.raises(ValueError):
        td.encode_controller(td.ControllerSample(record_toggle_count=0, session_id=0, connected=True, left=(0.0,) * 6))
    with pytest.raises(ValueError):
        td.controller_axes(ControllerInput(thumbstick=(0.0,)))


def test_recording_state_round_trip():
    for sample in (td.RecordingStateSample(recording=True, demo_index=4, stamp_ns=10), td.RecordingStateSample(recording=False)):
        frame_id, axes, buttons = td.encode_recording_state(sample)
        assert frame_id == td.RECORDING_STATE_FRAME_ID == "vr_link.recording.v1" and axes == []
        assert len(buttons) == td.RECORDING_BUTTON_COUNT == 2
        assert td.decode_recording_state(frame_id, axes, buttons, stamp_ns=sample.stamp_ns) == sample
    assert td.encode_recording_state(td.RecordingStateSample(recording=False))[2] == [0, -1]
    with pytest.raises(ValueError):
        td.decode_recording_state(td.RECORDING_STATE_FRAME_ID, [], [1])


def test_toggle_tracker_parity_within_a_session():
    tracker = td.RecordToggleTracker()
    assert not tracker.seeded
    assert tracker.update(_controller(count=7, session=1)) == 0  # first record seeds, never toggles
    assert tracker.seeded
    assert tracker.update(_controller(count=7, session=1)) == 0  # unchanged
    assert tracker.update(_controller(count=8, session=1)) == 1  # one press
    assert tracker.update(_controller(count=10, session=1)) == 0  # two presses skipped by sampling cancel out
    assert tracker.update(_controller(count=13, session=1)) == 1  # three presses -> net one toggle


def test_toggle_tracker_reseeds_on_a_new_session():
    tracker = td.RecordToggleTracker()
    assert tracker.update(_controller(count=7, session=1)) == 0
    assert tracker.update(_controller(count=8, session=1)) == 1
    # Headset reconnected: the counter restarts at 0. Without the session marker the jump
    # 8 -> 0 would be read as an odd number of presses. The new session only re-seeds.
    assert tracker.update(_controller(count=0, session=2)) == 0
    assert tracker.update(_controller(count=1, session=2)) == 1
    # A backwards jump inside one session (reconnect the relay did not notice) toggles by parity.
    assert tracker.update(_controller(count=0, session=2)) == 1
    tracker.reset()
    assert not tracker.seeded and tracker.update(_controller(count=5, session=2)) == 0
