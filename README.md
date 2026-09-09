# vr_link_schema

FlatBuffers wire contract for the VR pose/input uplink between the headset
application (`vr_system`, Unreal, C++) and the PC side (`hit-bench` simulator
and the real-robot producer, Python). This repository is the single source of
truth for that contract: consumers reference it as a git submodule and use
the pre-generated bindings committed under `generated/` — they never run
`flatc` themselves. It also ships the Python receiver package `vr_link`
(framing, record builder/decoder, TCP server-side device), see below.

## Channel layout

| Port | Direction | Format | Owner |
|------|-----------|--------|-------|
| 9100 | PC -> headset (video) | legacy `VSTR` binary records; **codec=5 payload is this schema's `SimStatus`** | framing NOT this repo / codec=5 payload this repo |
| 9101 | PC -> headset (calibration JSON, real robot only) | not this repo | robot camera server |
| 9102 | headset -> PC (pose/input) | **this schema** (`VRS1` framing) | this repo |

Port plan since 2026-09-09: the pose uplink moved from 9101 to 9102 so the
same port works for the simulator and the real robot, where 9101 is already
the calibration channel.

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

## Framing

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

## Code generation

The bindings are committed in this repository and regenerated here only:

```bash
scripts/generate.sh    # uses flatc on PATH, or FLATC=/path/to/flatc
```

- `generated/cpp/vr_link_generated.h` — included by `vr_system` (header-only,
  against its vendored flatbuffers C++ runtime; the consumer adds this repo's
  `generated/cpp` to its include paths).
- `generated/python/vr_link_generated.py` — the flatc Python binding (against
  the pip `flatbuffers` runtime). The repo is pip-installable
  (`pyproject.toml` exposes this module and the `vr_link` package): consumers
  run `pip install -e <submodule-path>` once in their environment setup and
  then `import vr_link_generated` / `import vr_link`. Re-run the install after
  pulling a pin that changes `pyproject.toml` (the editable layout changed
  when `vr_link` was added).

The script hard-fails unless flatc matches the pinned version (see
`FLATC_VERSION` in `scripts/generate.sh`). Bumping flatc means bumping
`vr_system`'s flatbuffers runtime submodule to the same tag in the same
change — its generated header static_asserts the version — and keeping
hit-bench's pip `flatbuffers` pin compatible.

## Python receiver package (`vr_link`)

Hand-written code under `python/vr_link/` (never under `generated/`):

- `vr_link.types` — decoded data model (`TrackedPose`, `ControllerInput`,
  `HandTrackingState`, `VRInputState`), the `VRDevice` interface, and the
  wire joint order `BODY_JOINT_NAMES` (alias `XROBOT_BODY_JOINT_NAMES`).
- `vr_link.tcp_device` — `VRS1_*` framing constants,
  `build_vr_state_record(...)` (test/sender-side packer),
  `decode_vr_state_payload(payload, *, timestamp, body_joint_adapter=None)`,
  and `TcpVRDevice(host="0.0.0.0", port=9102, stale_timeout=0.5,
  body_joint_adapter=None)`: a TCP server accepting one headset at a time,
  resynchronizing on the magic and keeping only the newest record;
  `advance(dt)` never blocks and reports `connected=False` when no fresh
  record exists.

Convention adapter contract: `body_joint_adapter` is a
`Callable[[str, Quat], Quat]` called once per body joint, in
`BODY_JOINT_NAMES` order, with the schema joint name and the wire wxyz
quaternion; it returns the wxyz quaternion the consumer wants stored. It is
never applied to `head` or hand poses. `None` (the default) returns the raw
wire values. The adapter is the consumer's calibration, not part of this
contract: hit-bench binds GMR's
`general_motion_retargeting.vr_uplink_convention.to_xrobot_convention`.

Tests (any environment with `flatbuffers` and `pytest`):

```bash
python -m pytest tests -q
```

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
