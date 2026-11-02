// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::metal {

/* Check if the Metal backend is available. */
MLX_API bool is_available();

/** Capture a GPU trace, saving it to an absolute file `path` */
MLX_API void start_capture(std::string path = "");
MLX_API void stop_capture();

/** Get information about the GPU and system settings. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

/* Set a custom path to mlx.metallib. Must be called before any MLX operation.
 */
MLX_API void set_metallib_path(const std::string& path);
MLX_API const std::string& get_metallib_path();

/**
 * Limit the GPU time that MLX uses in each frame, in milliseconds.
 *
 * Call `reset_gpu_time_budget()` once per frame. When the budget of the
 * current frame is used up, the thread that evaluates arrays waits for the
 * next frame. A value of 0 removes the limit.
 */
MLX_API void set_gpu_time_budget(double milliseconds);

/** Start a new frame for the GPU time budget. */
MLX_API void reset_gpu_time_budget();

} // namespace mlx::core::metal
