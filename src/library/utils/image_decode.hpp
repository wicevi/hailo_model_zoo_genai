/**
 * Copyright (c) 2019-2026 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
#pragma once

#include <cstdint>
#include <vector>

namespace hailo_ollama
{

struct DecodedFrame {
    std::vector<uint8_t> data; // RGB interleaved
    uint32_t width;
    uint32_t height;
};

DecodedFrame decode_and_resize_image(const std::vector<uint8_t> &encoded, uint32_t target_w, uint32_t target_h);

} // namespace hailo_ollama

