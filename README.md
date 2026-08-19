# vr_link_schema

FlatBuffers wire contract for the VR pose/input uplink between the headset
application (`vr_system`, Unreal, C++) and the simulation PC (`hit-bench`,
Python). This repository is the single source of truth for that contract:
both consumers reference it as a git submodule and use the pre-generated
bindings committed under `generated/` — they never run `flatc` themselves.

## Channel layout

| Port | Direction | Format | Owner |
|------|-----------|--------|-------|
| 9100 | sim -> headset (video) | legacy `VSTR` binary records | NOT this repo |
| 9101 | headset -> sim (pose/input) | **this schema** (`VRS1` framing), legacy `VPOS` v2 supported during migration | this repo |

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
  orientation convention adapter to these joints.
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
- `generated/python/vr_link_generated.py` — imported by `hit-bench` (against
  the pip `flatbuffers` runtime). The repo is pip-installable
  (`pyproject.toml` exposes the module): consumers run
  `pip install -e <submodule-path>` once in their environment setup and then
  `import vr_link_generated`.

The script hard-fails unless flatc matches the pinned version (see
`FLATC_VERSION` in `scripts/generate.sh`). Bumping flatc means bumping
`vr_system`'s flatbuffers runtime submodule to the same tag in the same
change — its generated header static_asserts the version — and keeping
hit-bench's pip `flatbuffers` pin compatible.

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
