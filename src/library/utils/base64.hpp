/**
 * Copyright (c) 2019-2026 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hailo_ollama
{

std::vector<uint8_t> base64_decode(const std::string &input);

} // namespace hailo_ollama

