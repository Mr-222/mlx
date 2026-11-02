// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <cmath>
#include <limits>

#include "mlx/backend/metal/gpu_time_budget.h"

namespace mlx::core::metal {

namespace {

// One-sided Chebyshev: P(x >= mean + k * std) <= 1 / (1 + k^2) = 5%.
constexpr double kBoundK = 4.36;
// Weight of a new sample in the running mean and variance.
constexpr double kAlpha = 0.1;
// Samples needed before the measured time per op is used.
constexpr int kWarmupSamples = 5;
// Time per op to use before there are enough samples.
constexpr double kDefaultOpSeconds = 30e-6;
// Few buffers in flight keep the feedback fast when the op cost changes.
constexpr int kMaxInflight = 2;
// Start a new frame if reset() is not called for this long.
constexpr auto kAutoResetPeriod = std::chrono::milliseconds(100);

constexpr int kNoLimit = std::numeric_limits<int>::max();

} // namespace

void GpuTimeBudget::set(double seconds) {
  std::lock_guard lk(mtx_);
  budget_ = std::max(seconds, 0.0);
  enabled_ = budget_ > 0;
  used_ = 0.0;
  frame_start_ = Clock::now();
  cv_.notify_all();
}

void GpuTimeBudget::reset() {
  std::lock_guard lk(mtx_);
  start_frame_locked();
  cv_.notify_all();
}

void GpuTimeBudget::start_frame_locked() {
  used_ = std::max(used_ - budget_, 0.0);
  frame_start_ = Clock::now();
}

double GpuTimeBudget::expected_op_seconds_locked() const {
  return samples_ < kWarmupSamples ? kDefaultOpSeconds : mean_;
}

int GpuTimeBudget::max_ops_locked() {
  if (budget_ <= 0) {
    return kNoLimit;
  }
  if (Clock::now() - frame_start_ > kAutoResetPeriod) {
    start_frame_locked();
  }
  if (inflight_ >= kMaxInflight) {
    return 0;
  }
  double op_bound = samples_ < kWarmupSamples
      ? kDefaultOpSeconds
      : mean_ + kBoundK * std::sqrt(var_);
  double ops = std::floor((budget_ - used_ - reserved_) / op_bound);
  if (ops >= 1) {
    return static_cast<int>(std::min(ops, double(kNoLimit)));
  }
  // Make progress if the bound for one op is larger than the budget.
  return (inflight_ == 0 && used_ < budget_) ? 1 : 0;
}

int GpuTimeBudget::max_ops() {
  std::lock_guard lk(mtx_);
  return max_ops_locked();
}

int GpuTimeBudget::wait_for_ops() {
  std::unique_lock lk(mtx_);
  int ops;
  while ((ops = max_ops_locked()) == 0) {
    cv_.wait_for(lk, kAutoResetPeriod);
  }
  return ops;
}

double GpuTimeBudget::reserve(int ops) {
  std::lock_guard lk(mtx_);
  double reserved = ops * expected_op_seconds_locked();
  reserved_ += reserved;
  inflight_++;
  return reserved;
}

void GpuTimeBudget::release(
    double reserved,
    int ops,
    double gpu_start,
    double gpu_end,
    bool measured) {
  std::lock_guard lk(mtx_);
  // Clear the rounding error when nothing is in flight.
  reserved_ = (--inflight_ == 0) ? 0.0 : std::max(reserved_ - reserved, 0.0);
  if (!measured) {
    // The span includes waits on other streams, so use the estimate.
    used_ += reserved;
    cv_.notify_all();
    return;
  }
  used_ += std::max(gpu_end - std::max(gpu_start, last_gpu_end_), 0.0);
  last_gpu_end_ = std::max(last_gpu_end_, gpu_end);
  double x = (gpu_end - gpu_start) / ops;
  if (samples_++ == 0) {
    mean_ = x;
  } else {
    double d = x - mean_;
    mean_ += kAlpha * d;
    var_ = (1 - kAlpha) * (var_ + kAlpha * d * d);
  }
  cv_.notify_all();
}

GpuTimeBudget& gpu_time_budget() {
  // Not destroyed at exit, because completion handlers can still run.
  static auto* budget = new GpuTimeBudget();
  return *budget;
}

} // namespace mlx::core::metal
