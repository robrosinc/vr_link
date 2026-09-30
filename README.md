# vr_link_schema

Wire contract and Python sender/receiver code for the link between the
headset application (`vr_system`, Unreal, C++) and the PC side (Isaac Sim in
`hit-bench`, MuJoCo, the real robot). This repository is the single source of
truth for that contract: consumers reference it as a git submodule and use
the pre-generated FlatBuffers bindings committed under `generated/` — they
never run `flatc` themselves. It also ships the Python package `vr_link`:
the VRS1 pose receiver, the VSTR video sender (server, H.264 encoder, frame
sources, standalone stream pump) and the stereo calibration tools.

## Channel layout

| Port | Direction | Format | Code |
|------|-----------|--------|------|
| 9100 | PC -> headset (video + records) | `VSTR` 48-byte header + payload (`vr_link.wire`); codec=1 H.264, 2 camera params JSON, 3/4 head/base pose, 5 `SimStatus` (this schema) | `vr_link.stream_server` |
| 9101 | PC -> headset (calibration JSON, on request) | `GET_CALIBRATION\n` -> JSON (`vr_link.calibration.schema`) | `vr_link.calibration.server` |
| 9102 | headset -> PC (pose/input) | **this schema** (`VRS1` framing) | `vr_link.tcp_device` |

Port plan since 2026-09-09: the pose uplink moved from 9101 to 9102 so the
same port works for the simulator and the real robot. Since 2026-09-29 the
simulators also answer on 9101, with an *ideal* calibration computed from
their live camera model, so the headset's calibration page works for every
target.

### Downlink status (`SimStatus`, VSTR codec=5)

New structured downlink data rides the existing VSTR channel as codec=5
records whose payload is the FlatBuffers table `robros.vr.SimStatus` (parse
with `GetRoot<SimStatus>`; `root_type` stays `VRState`). Framing, sequence
and timestamps come from the VSTR header. Like `VRState`, `SimStatus` always
carries full current state, never events. Legacy fixed-layout records
(codec=2 JSON, codec=3/4 pose) stay as they are; only new record types use
this schema.

### Recording toggle (`VRState.record_toggle_count`)

Headset-UI recording toggle requests travel as a **monotonic counter**, not
a one-frame pulse: receivers sample newest-only, so pulses can be lost. The
receiver seeds `last_seen` from the first record of a connection and applies
`(new - last_seen) % 2` toggles on change.

## Framing (VRS1)

FlatBuffers messages carry no length, so each record on the TCP byte stream
is preceded by an 8-byte big-endian prefix:

```
magic  u32 = 0x56525331 ("VRS1")
length u32 = FlatBuffers payload size in bytes (receivers reject > 65536)
payload    = FlatBuffers-encoded robros.vr.VRState
```

Receivers resynchronize on the magic after corrupt or partial data. On UDP
(possible future transport) the prefix is kept as a sanity check; one record
per datagram, no resynchronization needed.

## Data conventions

- Poses are in the **simulation world frame**: Z-up, right-handed, meters,
  quaternions in **wxyz** order. The sender (Unreal) converts from its native
  left-handed Z-up centimeter frame **before** transmission.
- `body_joints` holds exactly **24 entries** in the PICO body-tracking /
  SMPL-X body order (`Pelvis`, `Left_Hip`, `Right_Hip`, `Spine1`, `Left_Knee`,
  `Right_Knee`, `Spine2`, `Left_Ankle`, `Right_Ankle`, `Spine3`, `Left_Foot`,
  `Right_Foot`, `Neck`, `Left_Collar`, `Right_Collar`, `Head`,
  `Left_Shoulder`, `Right_Shoulder`, `Left_Elbow`, `Right_Elbow`,
  `Left_Wrist`, `Right_Wrist`, `Left_Hand`, `Right_Hand`). This matches the
  GMR "xrobot" retargeting source; the receiver applies its per-joint
  orientation convention adapter to these joints (Python: the
  `body_joint_adapter` hook of `vr_link`, see below). The wire itself carries
  the sender's bone-frame convention unchanged.
- `head` is the raw HMD pose for head teleoperation; **no** convention
  adapter is applied to it.
- `left_hand`/`right_hand` carry hand-tracking keypoints for finger
  teleoperation. `keypoints` holds exactly **26 entries** in the OpenXR
  `XR_EXT_hand_tracking` joint order (`Palm`, `Wrist`, `Thumb_Metacarpal`,
  `Thumb_Proximal`, `Thumb_Distal`, `Thumb_Tip`, then for each of
  `Index`/`Middle`/`Ring`/`Little`: `Metacarpal`, `Proximal`, `Intermediate`,
  `Distal`, `Tip`). Keypoint poses are **relative to the wrist keypoint**,
  expressed in the wrist frame (entry 1 is the identity pose); the
  world-frame anchor travels separately in `wrist`. `wrist` uses the
  hand-tracking runtime's wrist, which differs from the body-tracking wrist
  in `body_joints` by a per-hand yaw convention offset — receivers must not
  assume the two agree. `tracked` is the per-frame tracking flag; when it is
  false the keypoints are stale last-known poses and must not be acted on.
  The receiver performs all retargeting onto the robot hand; the sender only
  applies the axis handedness conversion.
- `VRState` always carries the full current state (never deltas or edge
  events). Button presses are transmitted as levels (`0.0`/`1.0`, or analog
  in `[0, 1]`) in every record, so a latest-record-wins receiver loses
  nothing when packets drop.
- Video frames are side-by-side stereo with the **right eye in the left
  half** (the hardware camera's `left_half: right` order); every sender
  (Isaac, MuJoCo, USB, DDS) reproduces it.

## Calibration contract (port 9101 / codec=2)

```json
{"image_width": 640, "image_height": 480,
 "left":  {"fx": .., "fy": .., "cx": .., "cy": .., "k1": .., "k2": .., "k3": .., "k4": .., "r": [9 floats]},
 "right": {...}}
```

Per-eye pinhole intrinsics in per-eye pixels, OpenCV **fisheye** coefficients
`k1..k4`, and the `stereoRectify` R1/R2 rotation (eye optical frame -> mid
frame, row-major). The receiver's screen shader always applies the fisheye
model, so a simulated pinhole camera is sent as the Taylor series of
tan(theta) (`k = 1/3, 2/15, 17/315, 62/2835`), never as zeros. Measured
hardware calibrations come from `vr-link-calibrate-cams`; simulators compute
theirs at request time (`vr_link.calibration.ideal`). The flat YAML file
format (`left_fx: ...`) is the on-disk form of the same data.

## Code generation

The bindings are committed in this repository and regenerated here only:

```bash
scripts/generate.sh    # uses flatc on PATH, or FLATC=/path/to/flatc
```

- `generated/cpp/vr_link_generated.h` — included by `vr_system` (header-only,
  against its vendored flatbuffers C++ runtime; the consumer adds this repo's
  `generated/cpp` to its include paths).
- `generated/python/vr_link_generated.py` — the flatc Python binding (against
  the pip `flatbuffers` runtime), exposed as the flat module
  `vr_link_generated` by `pyproject.toml`.

The script hard-fails unless flatc matches the pinned version (see
`FLATC_VERSION` in `scripts/generate.sh`). Bumping flatc means bumping
`vr_system`'s flatbuffers runtime submodule to the same tag in the same
change — its generated header static_asserts the version — and keeping
consumers' pip `flatbuffers` pin compatible.

## Python package (`vr_link`)

Hand-written code under `src/vr_link/` (never under `generated/`). The base
install needs only `flatbuffers` and `numpy`; everything heavier is an
optional extra and is imported lazily by the module that needs it:

| Extra | Pulls in | Enables |
|---|---|---|
| `video` | PyAV | `vr_link.encoder`, `vr_link.pump`, `vr-link-stream` |
| `yaml` | pyyaml | calibration YAML files, stream configs |
| `calibration` | OpenCV, pyyaml | `vr-link-capture-checkerboard`, `vr-link-calibrate-cams` |
| `dds` | (documentation only) | `vr_link.sources.dds` needs the workspace `igris_c_sdk` Python binding (py3.12 build, not pip-installable): install it in the consumer environment or set `IGRIS_C_SDK_ROOT` |

Consumers install editable from their submodule checkout (`pip install -e
<path>[video]`, or a uv/pixi path source) and re-run that install after
pulling a pin that changes `pyproject.toml`.

Modules:

- `vr_link.types`, `vr_link.tcp_device` — VRS1 uplink: decoded data model,
  the `VRDevice` interface, framing constants, `build_vr_state_record`,
  `decode_vr_state_payload(payload, *, timestamp, body_joint_adapter=None)`,
  and `TcpVRDevice(host="0.0.0.0", port=9102, stale_timeout=0.5,
  body_joint_adapter=None)`: a TCP server accepting one headset at a time,
  resynchronizing on the magic and keeping only the newest record;
  `advance(dt)` never blocks and reports `connected=False` when no fresh
  record exists.
- `vr_link.wire` — the VSTR header (`build_header`, `parse_header`, codec
  and flag constants) and `build_sim_status_payload`.
- `vr_link.stream_server` — `VstrTcpStreamServer`: non-blocking VSTR
  downlink (one headset at a time, per-connection writer queue, drops delta
  frames until a fresh connection's first IDR, disconnects a receiver that
  falls too far behind), camera parameters (codec=2), rendered head/base poses
  (codec=3/4) and `SimStatus` (codec=5) records. `vr_link.transport` is the
  transport interface.
- `vr_link.encoder` — `H264Encoder` (PyAV, Annex-B, one access unit per
  frame, SPS/PPS repeated on every IDR, `request_keyframe()` for new
  connections), `EncoderWorker` (worker thread that drops *unencoded* frames
  under back-pressure, for callers that must not pay encode time such as the
  Isaac control loop) and `resolve_h264_encoder(name)` (probe + libx264
  fallback; `libx264`, `h264_nvenc`, `h264_vaapi`, `h264_qsv` where the
  PyAV build and the host driver provide them).
- `vr_link.calibration` — `schema` (JSON contract, validation, tan-series
  `TAN_SERIES_K`), `ideal` (ideal stereo calibration from a rig description
  or from live camera intrinsics/poses, MuJoCo metadata adapter), `yaml_io`
  (flat YAML read/write), `server` (`CalibrationServer(provider, port=9101)`
  answering `GET_CALIBRATION` from a provider callable evaluated per request;
  providers: `yaml_file_provider`, `json_file_provider`, `static_provider`,
  `LatestValueProvider`).
- `vr_link.sources` — `Frame`, `StereoCameraMetadata`, the `FrameSource`
  protocol, and the sources `usb.UsbStereoSource` (V4L2/OpenCV, hardware tone
  tuning in `v4l2`), `mujoco_shm.MujocoShmSource` (igris_c_mujoco vision
  segment, seqlock reader, layout v1 and v2 with per-eye camera metadata),
  `dds.DdsStereoSource` (robot JPEG topic via igris_c_sdk).
- `vr_link.pump`, `vr_link.config`, `vr_link.cli.stream` — the standalone
  sender: `vr-link-stream --source usb|dds|mujoco [--config ...]` reads a
  `stream_*.yaml` (packaged defaults under `vr_link/data/configs/`), runs
  source -> encoder -> `VstrTcpStreamServer`, and serves calibration on 9101
  from `calibration.provider`: `yaml`/`json` (measured file, the real
  robot), `ideal_from_source` (MuJoCo segment metadata) or
  `ideal_from_config` (fovy/toe-out/baseline from the config).
- `vr_link.tools` — `vr-link-capture-checkerboard --source usb|dds
  --output-dir <dir>` and `vr-link-calibrate-cams --input-dir <captures>
  --output-dir <dir>` (fisheye or pinhole stereo calibration; writes the
  OpenCV FileStorage YAML, the flat Unreal YAML and the rectification maps).
  `vr-link-make-calibration` writes the ideal YAML of a simulated rig.
- `vr_link.data` — packaged configs (`configs/*.yaml`, tracked) and the
  calibration data directory `calibration/` (**git-ignored, local**):
  `igris_c_eyes_stereo_unreal.yaml` is the measured hardware calibration the
  USB/DDS senders serve by default, `mujoco_stereo_calibration.yaml` the
  generated ideal model. A fresh checkout has neither: run
  `vr-link-calibrate-cams` (or copy the robot's file) and
  `vr-link-make-calibration --output src/vr_link/data/calibration/mujoco_stereo_calibration.yaml`.
  Tests that read these files skip when they are absent.

Convention adapter contract: `body_joint_adapter` is a
`Callable[[str, Quat], Quat]` called once per body joint, in
`BODY_JOINT_NAMES` order, with the schema joint name and the wire wxyz
quaternion; it returns the wxyz quaternion the consumer wants stored. It is
never applied to `head` or hand poses. `None` (the default) returns the raw
wire values. The adapter is the consumer's calibration, not part of this
contract: hit-bench binds GMR's
`general_motion_retargeting.vr_uplink_convention.to_xrobot_convention`.

### Standalone use (pixi)

```bash
pixi install                       # default env: video + calibration + test
pixi run test                      # pytest (loopback and temp-file tests, no hardware)
pixi run mujoco-tcp-stream         # vr-link-stream --source mujoco
pixi run camera-tcp-stream         # vr-link-stream --source usb
pixi run jetson-tcp-stream         # vr-link-stream --source dds (igris_c_sdk required)
pixi run capture-checkerboard -- --output-dir captures
pixi run calibrate-cams -- --captures-root captures --output-dir calibration_out
pixi run make-mujoco-calibration -- --output mujoco_stereo_calibration.yaml
```

The conda-forge ffmpeg behind PyAV in this environment provides libx264,
h264_nvenc, h264_vaapi and h264_qsv; hardware encoders still need the host
driver at runtime. The pip PyAV wheel used by hit-bench provides libx264 only.

### MuJoCo vision segment

`igris_c_mujoco` renders `Left_Eye`/`Right_Eye` side by side into the POSIX
segment `/dev/shm/igris_c_mujoco_vision_shm` under a seqlock
(`igris_c_mujoco/include/igris_c_mujoco/vision_shm.h`). Layout v2 (128-byte
header) carries the per-eye camera model (fovy, pose in the parent body,
baseline, eye size) so `vr-link-stream --source mujoco` computes the ideal
calibration on request; a v1 producer (64-byte header) still streams, with
the calibration taken from the config's `ideal` section instead.

## Schema evolution rules

- **Add** fields only at the end of a `table`. Never reorder existing fields.
- **Never remove** a field; mark it `(deprecated)` instead.
- `struct`s (e.g. `Pose`) are frozen once shipped — they are fixed-size and
  cannot evolve. Extend via new table fields instead.
- Old readers skip unknown fields; new readers get type defaults for fields
  old senders omit. Consumers pin this repo at a commit, so the
  "schema version" of a build is the submodule pin.
- After changing `vr_link.fbs`, run `scripts/generate.sh` and commit the
  schema together with `generated/` in one commit, so pin = schema =
  bindings always agree. Consumers then only bump their submodule pin.
