"""GET_CALIBRATION request/response over loopback."""
import socket

import pytest

from vr_link.calibration import ideal, schema
from vr_link.calibration.server import CalibrationServer, LatestValueProvider, request_calibration, static_provider

CAL = ideal.ideal_stereo_from_fovy(640, 480, 90.0, 14.0, 52.152)


def test_request_returns_the_json_payload():
    with CalibrationServer(static_provider(CAL), host="127.0.0.1", port=0, request_timeout=1.0) as server:
        reply = request_calibration("127.0.0.1", server.port)
    cal = schema.decode_payload(reply)
    assert cal["left"]["fx"] == pytest.approx(240.0)
    assert "camera_model" not in cal
    assert server.served_count == 1


def test_not_ready_provider_closes_without_payload():
    provider = LatestValueProvider()
    with CalibrationServer(provider, host="127.0.0.1", port=0, request_timeout=1.0) as server:
        assert request_calibration("127.0.0.1", server.port) == b""
        provider.set(CAL)
        assert schema.decode_payload(request_calibration("127.0.0.1", server.port))["image_width"] == 640
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


def test_garbage_request_is_answered_after_timeout():
    with CalibrationServer(static_provider(CAL), host="127.0.0.1", port=0, request_timeout=0.2) as server:
        with socket.create_connection(("127.0.0.1", server.port), timeout=5.0) as conn:
            conn.sendall(b"HELLO")  # no newline: the server waits for the timeout, then answers anyway
            chunks = []
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
    assert schema.decode_payload(b"".join(chunks))["image_height"] == 480
