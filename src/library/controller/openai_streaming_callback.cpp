/**
 * Copyright (c) 2019-2026 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
#include "controller/openai_streaming_callback.hpp"

#include <algorithm>
#include <cstring>

#include <nlohmann/json.hpp>

namespace hailo_ollama
{

using json = nlohmann::ordered_json;

static std::string sse_line(const std::string &payload)
{
    return "data: " + payload + "\n\n";
}

OpenAIStreamingReadCallback::OpenAIStreamingReadCallback(const std::string &id, const std::string &model,
    hailort::genai::LLMGeneratorCompletion &&completion)
    : m_id(id),
      m_model(model),
      m_completion(std::move(completion)),
      m_sent_role(false),
      m_done(false),
      m_pending()
{
}

oatpp::v_io_size OpenAIStreamingReadCallback::read(void *buffer, v_buff_size bufferSize, oatpp::async::Action &action)
{
    (void)action;

    if (m_done && m_pending.empty()) {
        return 0;
    }

    if (m_pending.empty() && !m_done) {
        // Send initial role delta once (OpenAI streaming format)
        if (!m_sent_role) {
            json event = {
                {"id", m_id},
                {"object", "chat.completion.chunk"},
                {"created", static_cast<int64_t>(std::time(nullptr))},
                {"model", m_model},
                {"choices", json::array({{
                    {"index", 0},
                    {"delta", {{"role", "assistant"}}},
                    {"finish_reason", nullptr}
                }})}
            };
            m_pending = sse_line(event.dump());
            m_sent_role = true;
        } else {
            const auto status = m_completion.generation_status();
            if (status == hailort::genai::LLMGeneratorCompletion::Status::GENERATING) {
                const auto token = m_completion.read().expect("read failed!");
                json event = {
                    {"id", m_id},
                    {"object", "chat.completion.chunk"},
                    {"created", static_cast<int64_t>(std::time(nullptr))},
                    {"model", m_model},
                    {"choices", json::array({{
                        {"index", 0},
                        {"delta", {{"content", token}}},
                        {"finish_reason", nullptr}
                    }})}
                };
                m_pending = sse_line(event.dump());
            } else {
                const char *finish_reason =
                    (status == hailort::genai::LLMGeneratorCompletion::Status::LOGICAL_END_OF_GENERATION) ? "stop" : "length";
                json event = {
                    {"id", m_id},
                    {"object", "chat.completion.chunk"},
                    {"created", static_cast<int64_t>(std::time(nullptr))},
                    {"model", m_model},
                    {"choices", json::array({{
                        {"index", 0},
                        {"delta", json::object()},
                        {"finish_reason", finish_reason}
                    }})}
                };
                m_pending = sse_line(event.dump()) + sse_line("[DONE]");
                m_done = true;
            }
        }
    }

    const auto to_copy = static_cast<size_t>(std::min<v_buff_size>(bufferSize, static_cast<v_buff_size>(m_pending.size())));
    if (to_copy == 0) {
        return 0;
    }
    std::memcpy(buffer, m_pending.data(), to_copy);
    m_pending.erase(0, to_copy);
    return static_cast<oatpp::v_io_size>(to_copy);
}

} // namespace hailo_ollama
