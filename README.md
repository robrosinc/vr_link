# vr_link

Wire contract and Python bridge code for the link between the headset
application (`vr_system`, Unreal, C++) and the PC side (real robot NUC,
MuJoCo, later Isaac Sim). This repository is the single source of truth for
that contract: consumers reference it as a git submodule and use the
pre-generated FlatBuffers bindings committed under `generated/` — they never
run `flatc` themselves. It also ships the Python package `vr_link`: the VRS1
pose receiver, the VRL2 downlink sender (server, H.264 encoder, frame and
robot-state sources, DDS subscribers, timestamp matching, standalone pumps),
the simulator shared-memory contract and the stereo calibration tools.

**2.0 (breaking, downlink only):** the headset talks to `vr_link` only. The
VSTR 48-byte header and its codec records (JSON camera params, head/base
pose, `SimStatus`) are gone; every downlink record is a framed FlatBuffers
`Envelope`, and the robot state the headset used to fetch itself over
CycloneDDS now arrives frame-matched from the sender. The VRS1 uplink is
unchanged (byte-compatible with 1.x receivers such as `igris_c_gmr`).

## Channel layout

| Port | Direction | Format | Code |
|------|-----------|--------|------|
| 9100 | PC -> headset | `VRL2` framed `Envelope` stream: `Calibration` once per connection, then per frame `Signals` -> `RobotState` -> `VideoFrame` | `vr_link.network.tcp_server` |
| 9101 | headset -> PC request, PC -> headset reply | one framed `Envelope(CalibrationRequest)` in, one framed `Envelope(Calibration)` out, then close | `vr_link.network.calibration_server` |
| 9102 | headset -> PC | `VRS1` framed `VRState` (**unchanged since 1.x**) | `vr_link.network.tcp_device` |

## Framing

Both directions use the same 8-byte big-endian prefix; only the magic and
the root table differ:

```
uplink   (VRS1)  magic u32 = 0x56525331 "VRS1", length u32, payload = robros.vr.VRState   (root_type)
downlink (VRL2)  magic u32 = 0x56524C32 "VRL2", length u32, payload = robros.vr.Envelope  (GetRoot<Envelope>)
```

Receivers resynchronise on the magic after corrupt or partial data. VRS1
payloads are at most 65536 bytes; VRL2 payloads at most 8 MiB (the headset's
`MaxPacketBytes`).

## Downlink messages (`Envelope { schema_version=2, seq, wall_ns, sim_ns, body }`)

`seq` increases by one per Envelope on a connection. `wall_ns` is
CLOCK_REALTIME of the frame or sample, `sim_ns` the simulation time (0 for
real cameras). The `Signals` and `RobotState` records of a frame are sent
immediately before its `VideoFrame` with the same `wall_ns`/`sim_ns` and
consecutive `seq`; receivers pair them by that or by sender order.

| Body | Content |
|---|---|
| `VideoFrame` | one H.264 Annex-B access unit of the SBS frame (`codec=1`, `width`/`height` of the full SBS frame, `is_key`, `data`). Right eye in the LEFT half. SPS/PPS repeated on every IDR; nothing is sent to a new connection before its first IDR. |
| `Calibration` | per-eye `image_width`/`image_height` and `left`/`right` `CameraModel {fx fy cx cy k1 k2 k3 k4 r[9]}` (see "Calibration contract"). Sent once per 9100 connection (newest wins on the headset) and as the 9101 reply. |
| `CalibrationRequest` | empty; the 9101 request body. |
| `RobotState` | frame-matched joint state, four groups each with `*_valid` and `*_stamp_ns`: `body_actual[31]` (LowState), `body_reference[31]` (`gmr/refstate`), `hand_actual[12]` (HandState), `hand_command[12]` (`handcmd`), plus `base_pose: Pose` (valid only from simulators). An invalid group is an empty vector and must be ignored; a group the sender could not match within tolerance is invalid, never stale. |
| `Signals` | `recording` (episode recording active on the PC side) and `source: SourceKind {Unknown, Robot, MuJoCo, Isaac}` for the HUD label. Replaces 1.x `SimStatus`. |

### Joint orders

- `body_*`: 31 radians in the igris_c_sdk **joint-space** order (the order of
  `LowState.joint_state`, `gmr/refstate` and the MJCF serial actuators):
  `waist_yaw waist_roll waist_pitch`, left leg `hip_pitch hip_roll hip_yaw
  knee_pitch ankle_pitch ankle_roll`, right leg (same), left arm
  `shoulder_pitch shoulder_roll shoulder_yaw elbow_pitch wrist_yaw wrist_roll
  wrist_pitch`, right arm (same), `neck_yaw neck_pitch`
  (`vr_link.messages.robot_state.BODY_SERIAL_JOINT_NAMES`). The headset's FK
  indexes waist 0-2, arms 15-21 / 22-28, neck 29-30.
- `hand_*`: 12 values in motor-id order 11..16 (right: thumb flexion, index,
  middle, ring, little, thumb CMC) then 21..26 (left, same layout);
  normalized 0 = open, 1 = closed. `hand_command` is reordered by the wire
  `id`; `hand_actual` is taken positionally.

### Recording toggle (`VRState.record_toggle_count`)

Headset-UI recording toggle requests travel on the uplink as a **monotonic
counter**, not a one-frame pulse: receivers sample newest-only, so pulses can
be lost. The receiver seeds `last_seen` from the first record of a connection
and applies `(new - last_seen) % 2` toggles on change. The resulting state
comes back to the headset in `Signals.recording`.

## Uplink data conventions (`VRState`)

- Poses are in the **simulation world frame**: Z-up, right-handed, meters,
  quaternions in **wxyz** order. The sender (Unreal) converts from its native
  left-handed Z-up centimeter frame **before** transmission. `RobotState.base_pose`
  uses the same convention.
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
  world-frame anchor travels separately in `wrist`. `tracked` is the
  per-frame tracking flag; when it is false the keypoints are stale
  last-known poses and must not be acted on.
- `VRState` always carries the full current state (never deltas or edge
  events). Button presses are transmitted as levels (`0.0`/`1.0`, or analog
  in `[0, 1]`) in every record, so a latest-record-wins receiver loses
  nothing when packets drop.

## Calibration contract

`Calibration` carries, per eye, pinhole intrinsics in per-eye pixels, OpenCV
**fisheye** coefficients `k1..k4`, and the `stereoRectify` R1/R2 rotation
(eye optical frame -> mid frame, row-major). The receiver's screen shader
always applies the fisheye model, so a simulated pinhole camera is sent as
the Taylor series of tan(theta) (`k = 1/3, 2/15, 17/315, 62/2835`), never as
zeros. Measured hardware calibrations come from `vr-link-calibrate-cams`;
simulators compute theirs at request time (`vr_link.calibration.ideal`). The
flat YAML file (`left_fx: ...`) and the JSON file (`vr_link.calibration.schema`)
are on-disk forms of the same mapping.

## Simulator vision segment (SHM layout v3)

Simulators hand vr_link their frames and their robot state through one POSIX
shared-memory segment (`/dev/shm/igris_c_mujoco_vision_shm` for MuJoCo)
whose layout is owned here: `include/vr_link/vision_shm.h` (C++ producers
include it) mirrored by `vr_link.sources.shm` (`HEADER_V3_FORMAT`). The
producer writes under a seqlock (odd `seq` while writing); readers accept a
copy only if `seq` is even and unchanged.

| Offset | Content |
|---|---|
| 0..48 | v1: `magic` (`'MJV'<<8 \| version`), `seq`, `width`, `height`, `channels`, `step`, `stamp_realtime_ns`, `sim_time_ns`, `frame_index` |
| 48..128 | v2: static per-eye camera model (left/right `fovy_deg`, `pos[3]`, `quat_wxyz[4]`, `baseline_m`, `eye_width`, `eye_height`) -> ideal calibration on request |
| 128..384 | v3: robot state written with each frame: `state_stamp_realtime_ns`, `state_sim_time_ns`, `state_valid` bits (body_actual, hand_actual, base_pose, recording-flag-valid), `body_joint_count=31`, `hand_motor_count=12`, `state_flags` (recording), `body_actual[31]`, `hand_actual[12]`, `base_pose[7]` |
| 384.. | RGB8 pixels, SBS, right eye in the left half |

The reader accepts v1 (64 B), v2 (128 B) and v3 (384 B) producers; without a
v3 producer the SHM-sourced `RobotState` groups are simply invalid.

## Code generation

The bindings are committed in this repository and regenerated here only:

```bash
pixi run generate      # scripts/generate.sh with the pinned flatc from the pixi env
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
| `video` | PyAV | `vr_link.video.encoder`, `vr_link.pipeline`, `vr-link-stream` |
| `yaml` | pyyaml | calibration YAML files, stream configs |
| `calibration` | OpenCV, pyyaml | `vr-link-capture-checkerboard`, `vr-link-calibrate-cams`, the DDS camera source (JPEG decode) |
| `dds` | (marker only) | `vr_link.network.dds` and `vr_link.sources.dds_camera` need the `igris_c_sdk` Python binding: a py3.12 wheel a human builds from the `thirdparty/igris_c_sdk` submodule (`task wheel`) and installs with `pixi run install-sdk` (or `IGRIS_C_SDK_ROOT`). vr_link only subscribes. |

Consumers install editable from their submodule checkout (`pip install -e
<path>[video]`, or a uv/pixi path source) and re-run that install after
pulling a pin that changes `pyproject.toml`.

Layout:

- `vr_link.types`, `vr_link.network.tcp_device` (shim: `vr_link.tcp_device`) —
  VRS1 uplink: decoded data model, the `VRDevice` interface, framing
  constants, `build_vr_state_record`, `decode_vr_state_payload(payload, *,
  timestamp, body_joint_adapter=None)`, and `TcpVRDevice(host="0.0.0.0",
  port=9102, stale_timeout=0.5, body_joint_adapter=None)`: a TCP server
  accepting one headset at a time, resynchronising on the magic and keeping
  only the newest record; `advance(dt)` never blocks and reports
  `connected=False` when no fresh record exists.
- `vr_link.messages` — VRL2 framing (`frame_payload`, `RecordReader`,
  `parse_envelope`) and one module per body: `video`, `calibration`,
  `robot_state` (`RobotStateSample`, joint-order constants), `signals`.
- `vr_link.network.tcp_server` — `EnvelopeTcpServer`: one headset at a time,
  per-connection writer queue, `Calibration` once, `send_video(...,
  robot_state=, signals=)`; drops delta frames until a fresh connection's
  first IDR and disconnects a receiver that falls behind.
- `vr_link.network.calibration_server` — `CalibrationServer(provider,
  port=9101)` answering `CalibrationRequest` from a provider callable
  evaluated per request (`yaml_file_provider`, `json_file_provider`,
  `static_provider`, `LatestValueProvider`); `request_calibration()` client.
- `vr_link.network.dds` — `DdsChannel` (process-wide `ChannelFactory`),
  `StampedSubscription` + extractors for `LowState_`, `MasterArmState_`
  (`gmr/refstate`), `HandState_`, `HandCmd_`, and
  `start_state_subscriptions(...)`.
- `vr_link.utils.time_sync` — `StampedRingBuffer.nearest(target_ns,
  tolerance_ns)`: the frame/state matching the headset used to do.
- `vr_link.video.encoder` — `H264Encoder` (PyAV, Annex-B, SPS/PPS on every
  IDR, `request_keyframe()`), `EncoderWorker`, `resolve_h264_encoder(name)`.
- `vr_link.sources` — `Frame` (with `robot_block` from SHM v3),
  `StereoCameraMetadata`, `ShmRobotBlock`, the `FrameSource` protocol, and
  `usb.UsbStereoSource`, `shm.ShmSource` (v1..v3), `dds_camera.DdsCameraSource`
  (JPEG topic on a `DdsChannel`).
- `vr_link.pipeline` — `config` (YAML schema below), `pump.StreamPump`
  (source -> encoder -> server, with per-frame `RobotState`/`Signals`
  providers), `robot_state.RobotStateAssembler` (per-field `dds`/`shm`/`none`
  with tolerances 5/15/40/40 ms) and `SignalsProvider`.
- `vr_link.calibration` — `schema` (mapping validation, JSON file form,
  `TAN_SERIES_K`), `ideal` (ideal stereo model from a rig description or live
  camera intrinsics/poses, MuJoCo metadata adapter), `fov`, `yaml_io`.
- `vr_link.cli`, `vr_link.tools` — entry points (below) and the checkerboard
  capture / stereo calibration tools.
- `vr_link.configs` — the packaged configuration files (`stream_*.yaml`,
  `calibration_tools.yaml`; tracked, shipped in the wheel).
- `local/` (repository root, **git-ignored**) — machine-local data, never
  packaged: `captures/<yymmdd_HHMMSS>/` from `vr-link-capture-checkerboard`,
  `calibration/<capture>/` from `vr-link-calibrate-cams`, and the served
  measured calibration `calibration/igris_c_eyes_stereo_unreal.yaml`
  (`vr-link-calibrate-cams --serve-as igris_c_eyes_stereo_unreal.yaml` copies it
  there). The root is `$VR_LINK_LOCAL_DIR`, else `<checkout>/local`, else
  `~/.vr_link` for a wheel install (`vr_link.pipeline.config.local_root`). A
  fresh checkout has no served calibration: the 9101 server answers "not
  ready" until one exists. Tests that read these files skip when absent.

Convention adapter contract: `body_joint_adapter` is a
`Callable[[str, Quat], Quat]` called once per body joint, in
`BODY_JOINT_NAMES` order, with the schema joint name and the wire wxyz
quaternion; it returns the wxyz quaternion the consumer wants stored. It is
never applied to `head` or hand poses. `None` (the default) returns the raw
wire values. The adapter is the consumer's calibration, not part of this
contract: hit-bench binds GMR's
`general_motion_retargeting.vr_uplink_convention.to_xrobot_convention`.

### The sender: `vr-link-stream`

```
vr-link-stream --source usb   [--config stream_usb.yaml]  --dds-namespace igris_c_XXXX   NUC USB camera + robot DDS state
vr-link-stream --source dds   [--config stream_dds.yaml]  --dds-namespace igris_c_XXXX   igris_c_camera DDS topic + robot DDS state
vr-link-stream --source dds   --config stream_dds_mujoco.yaml --dds-namespace igris_c_XXXX  MuJoCo publishing the same DDS contract (ideal calibration from the file)
vr-link-stream --source shm   [--config stream_shm.yaml]                                simulator segment (video + state)
```

Config schema (`src/vr_link/configs/*.yaml`; `--config` >
`$VR_LINK_STREAM_CONFIG` > packaged default). Paths in a file resolve against
the file's directory; `pkg:` addresses the package (`pkg:configs/...`) and
`local:` the local data root (`local:calibration/...`):

```yaml
eyes_stereo: {cam_index, resolution: [1280, 480], fps, enable_stereo, apply_tuning}
tcp:         {host, port: 9100, bitrate, gop, fourcc, encoder: sw|libx264|h264_nvenc|h264_vaapi|h264_qsv}
calibration: {enable, host, port: 9101, provider: yaml|json|ideal_from_source|ideal_from_config,
              yaml, json, image_width, image_height, request_timeout, ideal: {fovy_deg, toe_out_deg, baseline_mm}}
dds:         {domain_id, namespace, topic_naming: native|ros_compatible, config_uri,
              camera_topic, frame_id, topics: {lowstate, handstate, handcmd, refstate}, buffer_capacity}
shm:         {shm_name, read_timeout_sec}
robot_state: {actual: dds|shm|none, reference: dds|none, hand_actual: dds|shm|none,
              hand_command: dds|none, base_pose: shm|none,
              tolerances_ms: {actual: 5, reference: 15, hand_actual: 40, hand_command: 40}}
signals:     {source: robot|mujoco|isaac|unknown, recording: shm|none}
```

Calibration providers: `yaml`/`json` serve a measured file (the real robot);
`ideal_from_source` computes the ideal model from the segment's camera
metadata (layout >= 2), falling back to `ideal_from_config`
(`calibration.ideal` + the source frame size).

### Standalone use (pixi)

```bash
pixi install                        # default env: video + calibration + test
pixi run test                       # pytest (loopback and temp-file tests, no hardware)
pixi run install-sdk                # igris_c_sdk wheel from thirdparty/igris_c_sdk/dist (human-built)
pixi run shm-tcp-stream             # vr-link-stream --source shm
pixi run camera-tcp-stream -- --dds-namespace igris_c_XXXX
pixi run dds-tcp-stream -- --dds-namespace igris_c_XXXX [--dds-domain N]   # igris_c_camera on <ns>/sensor/eyes_stereo/compressed
pixi run dds-tcp-stream-mujoco -- --dds-namespace igris_c_XXXX [--dds-domain N]   # igris_c_mujoco vision_dds_sink on the same topic
pixi run capture-checkerboard          # -> local/captures/<ts>/
pixi run capture-checkerboard-headless # same output, no window: robot NUC over SSH (stop igris-c-camera.service first), then scp the folder
pixi run focus-check                   # live per-eye sharpness (Laplacian variance) in the terminal: turn a lens ring until L and R agree
pixi run calibrate-cams -- --serve-as igris_c_eyes_stereo_unreal.yaml   # newest capture -> local/calibration/<ts>/ + served file
pixi run calibrate-cams -- --input-dir local/captures/<ts> --serve-as igris_c_eyes_stereo_unreal.yaml   # a folder copied from the NUC
pixi run make-mujoco-calibration       # -> local/calibration/mujoco_stereo_calibration.yaml (reference only; the shm stream computes its own)
```

The conda-forge ffmpeg behind PyAV in this environment provides libx264,
h264_nvenc, h264_vaapi and h264_qsv; hardware encoders still need the host
driver at runtime.

## Schema evolution rules

- **Add** fields only at the end of a `table`. Never reorder existing fields.
- **Never remove** a field; mark it `(deprecated)` instead. (2.0 removed the
  downlink-only `SimStatus` table as part of the VSTR -> VRL2 break; the
  uplink tables were not touched.)
- `struct`s (e.g. `Pose`) are frozen once shipped — they are fixed-size and
  cannot evolve. Extend via new table fields instead.
- Union members are appended only; `Envelope.schema_version` is bumped for
  incompatible downlink changes.
- Old readers skip unknown fields; new readers get type defaults for fields
  old senders omit. Consumers pin this repo at a commit, so the
  "schema version" of a build is the submodule pin.
- After changing `vr_link.fbs`, run `pixi run generate` and commit the
  schema together with `generated/` in one commit, so pin = schema =
  bindings always agree. Consumers then only bump their submodule pin.
