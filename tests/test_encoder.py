"""PyAV encoder: Annex-B access units, IDR + SPS/PPS on demand, metadata mapping."""
import threading

import numpy as np
import pytest

from vr_link.video import encoder as enc

av = pytest.importorskip("av")

W, H = 64, 48


def _frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def sw_available():
    if not enc.probe_h264_encoder(enc.SW_ENCODER, W, H):
        pytest.skip("libx264 not available in this PyAV build")


def test_parse_bitrate_and_aliases():
    assert enc.parse_bitrate(8_000_000) == 8_000_000
    assert enc.parse_bitrate("8M") == 8_000_000
    assert enc.parse_bitrate("800k") == 800_000
    assert enc.normalize_encoder_name("sw") == "libx264"
    assert enc.normalize_encoder_name("h264_nvenc") == "h264_nvenc"
    with pytest.raises(ValueError):
        enc.parse_bitrate("fast")


def test_annexb_nal_types_handles_both_start_code_lengths():
    au = b"\x00\x00\x00\x01\x67abc\x00\x00\x01\x68d\x00\x00\x01\x65ef"
    assert enc.annexb_nal_types(au) == [7, 8, 5]
    assert enc.is_idr_with_headers(au)
    assert not enc.is_idr_with_headers(b"\x00\x00\x01\x41xyz")


def test_first_access_unit_is_idr_with_headers(sw_available):
    with enc.H264Encoder(W, H, 30.0, codec="libx264", bitrate="200k", gop=30) as encoder:
        units = encoder.encode(_frame(0), user={"n": 0})
        assert len(units) == 1
        first = units[0]
        assert first.is_key
        assert enc.is_idr_with_headers(first.data)
        assert first.user == {"n": 0}
        deltas = [encoder.encode(_frame(i), user=i)[0] for i in range(1, 5)]
        assert [u.user for u in deltas] == [1, 2, 3, 4]
        assert not any(u.is_key for u in deltas)


def test_keyframe_request_mid_stream_yields_idr(sw_available):
    with enc.H264Encoder(W, H, 30.0, codec="libx264", bitrate="200k", gop=300) as encoder:
        encoder.encode(_frame(0))
        for i in range(1, 4):
            encoder.encode(_frame(i))
        encoder.request_keyframe()
        forced = encoder.encode(_frame(9))[0]
        assert forced.is_key
        assert enc.is_idr_with_headers(forced.data)
        after = encoder.encode(_frame(10))[0]
        assert not after.is_key


def test_encoder_rejects_wrong_shape(sw_available):
    with enc.H264Encoder(W, H, 30.0, codec="libx264") as encoder:
        with pytest.raises(ValueError):
            encoder.encode(np.zeros((H, W + 2, 3), dtype=np.uint8))


def test_resolve_falls_back_to_libx264(sw_available):
    assert enc.resolve_h264_encoder("no_such_encoder_xyz") == "libx264"


def test_worker_delivers_in_order_and_applies_keyframe_requests(sw_available):
    received = []
    done = threading.Event()

    def on_output(unit: enc.EncodedAccessUnit) -> None:
        received.append(unit)
        if unit.user == 5:
            done.set()

    worker = enc.EncoderWorker(enc.H264Encoder(W, H, 30.0, codec="libx264", gop=300), on_output, max_pending=8)
    try:
        for i in range(3):
            assert worker.submit(_frame(i), user=i)
        worker.request_keyframe()
        for i in range(3, 6):
            worker.submit(_frame(i), user=i)
        assert done.wait(5.0)
    finally:
        worker.close()
    assert [u.user for u in received] == [0, 1, 2, 3, 4, 5]
    assert received[0].is_key and received[3].is_key
    assert not received[1].is_key and not received[4].is_key
