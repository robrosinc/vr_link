"""Per-message builders/decoders: RobotState, Calibration, VideoFrame."""
import numpy as np
import pytest

from vr_link.calibration import ideal, schema
from vr_link.messages import (
    BODY_JOINT_COUNT,
    BODY_SERIAL_JOINT_NAMES,
    HAND_MOTOR_COUNT,
    HAND_MOTOR_IDS,
    MessageKind,
    RobotStateSample,
    build_calibration_payload,
    build_robot_state_payload,
    build_video_frame_payload,
    decode_calibration,
    decode_robot_state,
    decode_video_frame,
    parse_envelope,
    VideoStream,
)
from vr_link.messages.robot_state import vector


def test_joint_order_contract_constants():
    assert BODY_JOINT_COUNT == 31 and len(BODY_SERIAL_JOINT_NAMES) == 31
    assert BODY_SERIAL_JOINT_NAMES[:3] == ("waist_yaw", "waist_roll", "waist_pitch")
    assert BODY_SERIAL_JOINT_NAMES[15] == "l_shoulder_pitch" and BODY_SERIAL_JOINT_NAMES[22] == "r_shoulder_pitch"
    assert BODY_SERIAL_JOINT_NAMES[29:] == ("neck_yaw", "neck_pitch")
    assert HAND_MOTOR_COUNT == 12 and HAND_MOTOR_IDS == (11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26)


def test_robot_state_round_trip_with_all_groups():
    sample = RobotStateSample(
        body_actual=vector(np.arange(31) * 0.01, 100),
        body_reference=vector(np.arange(31) * 0.02, 101),
        hand_actual=vector(np.linspace(0, 1, 12), 102),
        hand_command=vector(np.linspace(1, 0, 12), 103),
        base_pose=(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0),
    )
    env = parse_envelope(build_robot_state_payload(sample, seq=3, wall_ns=10, sim_ns=20))
    assert env.kind is MessageKind.ROBOT_STATE
    out = decode_robot_state(env)
    for name in ("body_actual", "body_reference", "hand_actual", "hand_command"):
        got, want = getattr(out, name), getattr(sample, name)
        assert got.stamp_ns == want.stamp_ns
        assert got.values.dtype == np.float32
        np.testing.assert_allclose(got.values, want.values, rtol=1e-6)
    assert out.base_pose == pytest.approx(sample.base_pose)


def test_robot_state_missing_groups_decode_as_none():
    sample = RobotStateSample(body_actual=vector(np.zeros(31), 5))
    out = decode_robot_state(parse_envelope(build_robot_state_payload(sample, seq=0, wall_ns=0)))
    assert out.body_actual is not None and out.body_actual.stamp_ns == 5
    assert out.body_reference is None and out.hand_actual is None and out.hand_command is None and out.base_pose is None
    assert RobotStateSample().is_empty() and not sample.is_empty()


def test_robot_state_rejects_wrong_lengths_and_bad_pose():
    with pytest.raises(ValueError):
        build_robot_state_payload(RobotStateSample(body_actual=vector(np.zeros(30))), seq=0, wall_ns=0)
    with pytest.raises(ValueError):
        build_robot_state_payload(RobotStateSample(hand_command=vector(np.zeros(13))), seq=0, wall_ns=0)
    with pytest.raises(ValueError):
        build_robot_state_payload(RobotStateSample(base_pose=(1.0, 2.0)), seq=0, wall_ns=0)
    with pytest.raises(ValueError):
        build_robot_state_payload(RobotStateSample(body_actual=vector([float("nan")] * 31)), seq=0, wall_ns=0)


def test_calibration_round_trip_matches_validated_mapping():
    cal = ideal.ideal_stereo_from_fovy(640, 480, 90.0, 14.0, 52.152)
    env = parse_envelope(build_calibration_payload(cal, seq=1, wall_ns=2))
    assert env.kind is MessageKind.CALIBRATION
    out = decode_calibration(env)
    want = schema.strip_meta(schema.validate_calibration(cal))
    assert (out["image_width"], out["image_height"]) == (640, 480)
    assert ideal.flatten_sides(out) == pytest.approx(ideal.flatten_sides(want), rel=1e-6)
    assert len(out["left"]["r"]) == 9


def test_calibration_builder_validates_input():
    with pytest.raises(schema.CalibrationError):
        build_calibration_payload({"image_width": 640}, seq=0, wall_ns=0)


def test_video_frame_round_trip():
    au = bytes(range(256)) * 4
    env = parse_envelope(build_video_frame_payload(au, is_key=True, width=1280, height=480, seq=11, wall_ns=7, sim_ns=8))
    assert env.kind is MessageKind.VIDEO_FRAME and (env.seq, env.wall_ns, env.sim_ns) == (11, 7, 8)
    frame = decode_video_frame(env)
    assert (frame.codec, frame.width, frame.height, frame.is_key) == (1, 1280, 480, True)
    assert frame.data == au
    with pytest.raises(ValueError):
        build_video_frame_payload(b"", is_key=False, width=1, height=1, seq=0, wall_ns=0)
    assert frame.stream_id is VideoStream.EYES_STEREO  # the default, what pre-2.2 senders produce


def test_video_frame_stream_id_round_trips_and_rejects_unknown_streams():
    env = parse_envelope(build_video_frame_payload(b"au", is_key=False, width=640, height=480, seq=1, wall_ns=2, stream_id=VideoStream.HEAD_RGB))
    frame = decode_video_frame(env)
    assert frame.stream_id is VideoStream.HEAD_RGB and frame.stream_id == 1 and (frame.width, frame.height) == (640, 480)
    assert decode_video_frame(parse_envelope(build_video_frame_payload(b"au", is_key=True, width=1, height=1, seq=0, wall_ns=0, stream_id=0))).stream_id is VideoStream.EYES_STEREO
    with pytest.raises(ValueError):
        build_video_frame_payload(b"au", is_key=True, width=1, height=1, seq=0, wall_ns=0, stream_id=7)
