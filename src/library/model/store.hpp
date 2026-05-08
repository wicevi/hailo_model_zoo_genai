/**
 * Copyright (c) 2019-2026 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file store.hpp
 * @brief Interface for managing models
 **/

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "utils/interface.hpp"

namespace hailo_ollama
{

enum class ModelType {
    LLM,
    VLM
};

struct ModelInfo {
    std::string name;
    std::string hef_resource;
    std::string details;
    std::string license;
    ModelType type = ModelType::LLM;
    uint32_t frame_width = 0;   // VLM only
    uint32_t frame_height = 0;  // VLM only
};

class ModelStore : Interface
{
public:
    virtual std::optional<ModelInfo> get_model(const std::string &name) = 0;
    virtual std::vector<std::string> get_model_names() = 0;
};

} // namespace hailo_ollama
