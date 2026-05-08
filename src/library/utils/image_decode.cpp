/**
 * Copyright (c) 2019-2026 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
#include "utils/image_decode.hpp"

#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#include <stb_image_resize2.h>

namespace hailo_ollama
{

DecodedFrame decode_and_resize_image(const std::vector<uint8_t> &encoded, uint32_t target_w, uint32_t target_h)
{
    if (encoded.empty()) {
        throw std::runtime_error("decode_and_resize_image: empty input");
    }
    if (target_w == 0 || target_h == 0) {
        throw std::runtime_error("decode_and_resize_image: invalid target size");
    }

    int w = 0, h = 0, channels = 0;
    stbi_uc *decoded = stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()), &w, &h, &channels, 3);
    if (!decoded) {
        throw std::runtime_error("decode_and_resize_image: stbi_load_from_memory failed");
    }

    std::vector<uint8_t> resized(target_w * target_h * 3);
    // stb_image_resize2 easy API returns output pointer (same as output_pixels) or nullptr on failure
    unsigned char *const resized_ptr = stbir_resize_uint8_linear(decoded, w, h, 0,
        resized.data(), static_cast<int>(target_w), static_cast<int>(target_h), 0,
        static_cast<stbir_pixel_layout>(3));
    stbi_image_free(decoded);

    if (!resized_ptr) {
        throw std::runtime_error("decode_and_resize_image: resize failed");
    }

    return DecodedFrame{std::move(resized), target_w, target_h};
}

} // namespace hailo_ollama

