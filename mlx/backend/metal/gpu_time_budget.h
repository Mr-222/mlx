// Copyright © 2026 Apple Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace mlx::core::metal {

// Limits the GPU time that MLX uses in each frame.
//
// The app calls reset() once per frame. MLX charges the GPU time of each
// completed command buffer to the current frame. A new command buffer gets
// only the number of ops that fit in the remaining time. The time per op comes
// from a running mean and variance, and the one-sided Chebyshev bound keeps
// the chance of an overrun at about 5%.
class GpuTimeBudget {
 public:
  // Set the budget in seconds. 0 removes the limit.
  void set(double seconds);

  // Start a new frame. Time used above the budget in the last frame is
  // charged to the new frame.
  void reset();

  bool enabled() const {
    return enabled_.load(std::memory_order_relaxed);
  }

  // The max number of ops for a new command buffer. 0 means there is no
  // room now.
  int max_ops();

  // Wait until there is room, then return the max number of ops.
  int wait_for_ops();

  // Reserve the expected time of a command buffer before it is committed.
  // Call release() with the returned value when the buffer completes.
  double reserve(int ops);

  // Charge a completed command buffer, with its GPU start and end times in
  // seconds. Use `measured = false` if the buffer waited on other streams.
  void release(
      double reserved,
      int ops,
      double gpu_start,
      double gpu_end,
      bool measured);

 private:
  using Clock = std::chrono::steady_clock;

  void start_frame_locked();
  int max_ops_locked();
  double expected_op_seconds_locked() const;

  std::mutex mtx_;
  std::condition_variable cv_;
  std::atomic<bool> enabled_{false};
  double budget_{0.0};
  double used_{0.0};
  double reserved_{0.0};
  int inflight_{0};
  // GPU end time of the latest completed buffer. Overlapping buffers are
  // charged once.
  double last_gpu_end_{0.0};
  Clock::time_point frame_start_;

  // Running mean and variance of the GPU time per op.
  double mean_{0.0};
  double var_{0.0};
  int samples_{0};
};

GpuTimeBudget& gpu_time_budget();

} // namespace mlx::core::metal
