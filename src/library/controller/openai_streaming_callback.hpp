/**
 * Copyright (c) 2019-2026 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <hailo/genai/llm/llm.hpp>
#include <oatpp/data/stream/Stream.hpp>

namespace hailo_ollama
{

class OpenAIStreamingReadCallback : public oatpp::data::stream::ReadCallback
{
public:
    OpenAIStreamingReadCallback(const std::string &id, const std::string &model,
        hailort::genai::LLMGeneratorCompletion &&completion);

    oatpp::v_io_size read(void *buffer, v_buff_size bufferSize, oatpp::async::Action &action) override;

private:
    std::string m_id;
    std::string m_model;
    hailort::genai::LLMGeneratorCompletion m_completion;
    bool m_sent_role;
    bool m_done;
    std::string m_pending;
};

} // namespace hailo_ollama
