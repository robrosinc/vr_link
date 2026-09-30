// vr_link vision shared-memory contract (layout v3).
//
// A simulator (igris_c_mujoco today, Isaac Sim later) renders the side-by-side
// stereo frame (right eye in the LEFT half, like the hardware) into a POSIX
// shared-memory segment and vr_link (`vr_link.sources.shm`) reads it. This
// header is the single definition of the layout; the Python reader mirrors it
// with struct format "<IIIIIIQQQff3f4f3f4ffII4xQQIIII31f12f7f24x" (384 bytes).
//
// Seqlock: the producer stores seq = odd, writes header state + pixels, then
// stores seq = even (release). Readers copy while seq is even and accept the
// copy only if seq is unchanged afterwards. magic is stored last on attach and
// cleared (0) on detach. The static camera model (48..128) is written once
// before magic; the robot-state block (128..384) is written under the seqlock
// with every frame so it is exactly the state of the rendered scene.
//
// Layout versions share the magic base 'MJV'<<8 with the version in the low
// byte; readers accept v1 (64 B), v2 (128 B) and v3 (384 B) headers.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vr_link::vision_shm {

constexpr uint32_t kMagicBase = 0x4D4A5600u;  // 'MJV' << 8, low byte = layout version
constexpr uint32_t kLayoutVersion = 3;
constexpr uint32_t kMagic = kMagicBase | kLayoutVersion;

constexpr uint32_t kBodyJointCount = 31;  // igris_c_sdk joint-space serial order
constexpr uint32_t kHandMotorCount = 12;  // motor ids 11..16 (right), 21..26 (left)

constexpr std::size_t kHeaderBytesV1 = 64;
constexpr std::size_t kHeaderBytesV2 = 128;
constexpr std::size_t kHeaderBytesV3 = 384;

// state_valid bits
constexpr uint32_t kStateValidBodyActual = 1u << 0;
constexpr uint32_t kStateValidHandActual = 1u << 1;
constexpr uint32_t kStateValidBasePose = 1u << 2;
constexpr uint32_t kStateValidRecording = 1u << 3;  // state_flags.recording is meaningful
// state_flags bits
constexpr uint32_t kStateFlagRecording = 1u << 0;

struct HeaderV3 {
  // ---- v1 frame header (0..48) ----
  std::atomic<uint32_t> magic;   //   0  kMagic while a producer is attached, 0 otherwise
  std::atomic<uint32_t> seq;     //   4  seqlock: odd while header state + pixels are being written
  uint32_t width;                //   8  SBS frame width
  uint32_t height;               //  12
  uint32_t channels;             //  16  3 (RGB8)
  uint32_t step;                 //  20  width * channels
  uint64_t stamp_realtime_ns;    //  24  CLOCK_REALTIME at capture
  uint64_t sim_time_ns;          //  32
  uint64_t frame_index;          //  40
  // ---- v2 static camera model (48..128), written once before magic ----
  float left_fovy_deg;           //  48
  float right_fovy_deg;          //  52
  float left_pos[3];             //  56  parent-body frame, metres
  float left_quat_wxyz[4];       //  68
  float right_pos[3];            //  84
  float right_quat_wxyz[4];      //  96
  float baseline_m;              // 112
  uint32_t eye_width;            // 116
  uint32_t eye_height;           // 120
  uint8_t reserved_v2[4];        // 124
  // ---- v3 robot state block (128..384), written under the seqlock with the pixels ----
  uint64_t state_stamp_realtime_ns;        // 128  CLOCK_REALTIME when the state was sampled
  uint64_t state_sim_time_ns;              // 136
  uint32_t state_valid;                    // 144  kStateValid* bits
  uint32_t body_joint_count;               // 148  kBodyJointCount
  uint32_t hand_motor_count;               // 152  kHandMotorCount
  uint32_t state_flags;                    // 156  kStateFlag* bits
  float body_actual[kBodyJointCount];      // 160  rad
  float hand_actual[kHandMotorCount];      // 284  normalized 0 (open) .. 1 (closed)
  float base_pose[7];                      // 332  px py pz qw qx qy qz, sim world frame (Z-up, m)
  uint8_t reserved_v3[24];                 // 360
};                                         // 384 = kHeaderBytesV3

static_assert(std::is_standard_layout_v<HeaderV3>, "HeaderV3 must be standard layout");
static_assert(offsetof(HeaderV3, left_fovy_deg) == 48, "v2 block offset");
static_assert(offsetof(HeaderV3, state_stamp_realtime_ns) == 128, "v3 block offset");
static_assert(offsetof(HeaderV3, body_actual) == 160, "body_actual offset");
static_assert(offsetof(HeaderV3, hand_actual) == 284, "hand_actual offset");
static_assert(offsetof(HeaderV3, base_pose) == 332, "base_pose offset");
static_assert(sizeof(HeaderV3) == kHeaderBytesV3, "HeaderV3 size");

// Whole segment: header followed by the pixels (no padding: 384 is 8-aligned).
template <std::size_t FrameBytes>
struct Segment {
  HeaderV3 header;
  uint8_t pixels[FrameBytes];
};
static_assert(offsetof(Segment<16>, pixels) == kHeaderBytesV3, "pixels offset");

}  // namespace vr_link::vision_shm
