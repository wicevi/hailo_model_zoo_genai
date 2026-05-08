/**
 * Copyright (c) 2019-2026 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file generation_context.hpp
 * @brief Class for managing generation
 **/

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <cstdint>
#include <string>
#include <string_view>

#include <hailo/genai/llm/llm.hpp>
#include <hailo/genai/vlm/vlm.hpp>
#include <hailo/vdevice.hpp>
#include <libguarded/cs_plain_guarded.h>

#include "model/store.hpp"

namespace hailo_ollama
{

struct GenerationOptions {
    std::optional<float> temperature;
    std::optional<int32_t> seed;
    std::optional<uint32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> frequency_penalty;
    std::optional<uint32_t> num_predict;
};

struct Generation {
    std::string model_name;
    std::filesystem::path model_path;
    std::vector<std::string> prompt_json_strings;
    // LLM only: SDK may not allow default-constructing LLMGeneratorParams.
    std::optional<hailort::genai::LLMGeneratorParams> generator_params;
    GenerationOptions options;
    std::optional<std::chrono::seconds> keep_alive;
    ModelType model_type = ModelType::LLM;
    std::vector<std::vector<uint8_t>> image_buffers; // VLM: decoded RGB frames
    std::vector<std::string> messages_json;          // VLM: structured JSON messages
};

// We want a unique_ptr for LLM but we can't have it so this wrapper is needed
class LLMWrapper
{
public:
    explicit LLMWrapper(hailort::genai::LLM &&llm);

    hailort::genai::LLM &operator*();
    hailort::genai::LLM *operator->();

private:
    hailort::genai::LLM m_llm;
};

class VLMWrapper
{
public:
    explicit VLMWrapper(hailort::genai::VLM &&vlm);

    hailort::genai::VLM &operator*();
    hailort::genai::VLM *operator->();

private:
    hailort::genai::VLM m_vlm;
};

enum class ExpiryWaitStatus {
    SUCCESS,
    STOP
};

class GenerationContext
{
public:
    GenerationContext();
    explicit GenerationContext(std::optional<std::string> vdevice_group_id);
    hailort::genai::LLMGeneratorCompletion generate_one(const Generation &params);
    hailort::genai::LLMGeneratorCompletion generate_one_vlm(const Generation &params);

    void append_assistant_message(const std::string &content);

    std::string get_model_name() const;

    std::chrono::steady_clock::time_point get_expiration() const;

    std::string get_generation_recovery_sequence() const;
    hailort::genai::LLMGeneratorParams create_generator_params() const;

    void load_model(const std::string &model_name, std::filesystem::path model_path,
        std::optional<std::chrono::seconds> keep_alive, ModelType model_type = ModelType::LLM);
    void reset();

    ExpiryWaitStatus wait_expiry(std::unique_lock<std::mutex> &lock);

    void stop();

    struct GenerationFinishedReason {
        static constexpr const char *STOP = "stop";
        static constexpr const char *LENGTH = "length";
    };

private:
    std::optional<std::string> m_vdevice_group_id;
    std::shared_ptr<hailort::VDevice> m_vdevice;
    std::unique_ptr<LLMWrapper> m_llm;
    std::unique_ptr<VLMWrapper> m_vlm;
    ModelType m_loaded_type = ModelType::LLM;
    std::string m_model_name;
    std::filesystem::path m_last_path;
    std::vector<std::string> m_conversation_history; // Track conversation as JSON messages
    std::chrono::steady_clock::time_point m_last_generation;
    std::optional<std::chrono::seconds> m_keep_alive;
    std::condition_variable m_keep_alive_shortened;
    bool m_stop_flag;
};

using SyncGenerationContext = libguarded::plain_guarded<GenerationContext>;

} // namespace hailo_ollama
