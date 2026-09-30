"""Calibration request/response (framed Envelopes) over loopback."""
import socket

import pytest

from vr_link.calibration import ideal, schema
from vr_link.messages import RecordReader, decode_calibration, parse_envelope
from vr_link.network.calibration_server import CalibrationServer, LatestValueProvider, request_calibration, static_provider

CAL = ideal.ideal_stereo_from_fovy(640, 480, 90.0, 14.0, 52.152)


def test_request_returns_a_decoded_calibration():
    with CalibrationServer(static_provider(CAL), host="127.0.0.1", port=0, request_timeout=1.0) as server:
        cal = request_calibration("127.0.0.1", server.port)
    assert cal["left"]["fx"] == pytest.approx(240.0)
    assert (cal["image_width"], cal["image_height"]) == (640, 480)
    assert "camera_model" not in cal
    assert server.served_count == 1


def test_not_ready_provider_closes_without_payload():
    provider = LatestValueProvider()
    with CalibrationServer(provider, host="127.0.0.1", port=0, request_timeout=1.0) as server:
        with pytest.raises(schema.CalibrationError):
            request_calibration("127.0.0.1", server.port)
        provider.set(CAL)
        assert request_calibration("127.0.0.1", server.port)["image_width"] == 640
    assert server.failed_count == 1 and server.served_count == 1


def test_provider_is_called_per_request():
    calls = []

    def provider():
        calls.append(1)
        return CAL

    with CalibrationServer(provider, host="127.0.0.1", port=0, request_timeout=1.0) as server:
        request_calibration("127.0.0.1", server.port)
        request_calibration("127.0.0.1", server.port)
    assert len(calls) == 2


def _read_all(conn):
    chunks = []
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def test_legacy_text_request_is_answered_after_timeout():
    with CalibrationServer(static_provider(CAL), host="127.0.0.1", port=0, request_timeout=0.2) as server:
        with socket.create_connection(("127.0.0.1", server.port), timeout=5.0) as conn:
            conn.sendall(b"GET_CALIBRATION\n")  # 1.x request: not a frame, answered anyway
            reply = _read_all(conn)
    records = RecordReader().feed(reply)
    assert len(records) == 1
    assert decode_calibration(parse_envelope(records[0]))["image_height"] == 480


def test_bare_connect_is_answered_after_timeout():
    with CalibrationServer(static_provider(CAL), host="127.0.0.1", port=0, request_timeout=0.2) as server:
        with socket.create_connection(("127.0.0.1", server.port), timeout=5.0) as conn:
            reply = _read_all(conn)
    assert decode_calibration(parse_envelope(RecordReader().feed(reply)[0]))["left"]["fx"] == pytest.approx(240.0)
