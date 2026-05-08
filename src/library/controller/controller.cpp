/**
 * Copyright (c) 2019-2026 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file controller.cpp
 * @brief MyController implementation
 **/

#include "controller.hpp"

#include <cstdint>
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>

#include <hailo/genai/llm/llm.hpp>
#include <nlohmann/json.hpp>
#include <oatpp/base/Log.hpp>
#include <oatpp/data/mapping/ObjectMapper.hpp>
#include <oatpp/macro/codegen.hpp>
#include <oatpp/macro/component.hpp>
#include <oatpp/web/protocol/http/outgoing/StreamingBody.hpp>
#include <oatpp/web/server/api/ApiController.hpp>

#include "config/static_config.hpp"
#include "controller/llm_generation_callback.hpp"
#include "controller/openai_streaming_callback.hpp"
#include "controller/pull_callback.hpp"
#include "dto/DTOs.hpp"
#include "generation_context/generation_context.hpp"
#include "model/resource.hpp"
#include "model/store.hpp"
#include "oatpp/Types.hpp"
#include "utils/base64.hpp"
#include "utils/image_decode.hpp"
#include "utils/time.hpp"

namespace fs = std::filesystem;

// manage all long imports  from oatpp
namespace oat
{
using OutgoingResponse = oatpp::web::protocol::http::outgoing::Response;
using OutgoingStreamingBody = oatpp::web::protocol::http::outgoing::StreamingBody;
} // namespace oat

namespace hailo_ollama
{

using json = nlohmann::ordered_json;

static bool debug_requests_enabled()
{
    return (std::getenv("HAILO_OLLAMA_DEBUG_REQUESTS") != nullptr);
}

static bool webui_compat_enabled()
{
    // OpenWebUI may only enable vision for a whitelist of families/formats.
    // When enabled, we spoof VLM models' details to match common Ollama vision models.
    return (std::getenv("HAILO_OLLAMA_OPENWEBUI_COMPAT") != nullptr);
}

static std::string truncate_for_log(const std::string &s, size_t max_len)
{
    if (s.size() <= max_len) {
        return s;
    }
    return s.substr(0, max_len) + "...(truncated, len=" + std::to_string(s.size()) + ")";
}

static json redact_images_in_openai_body(const json &root)
{
    // Redact large base64 payloads while keeping structure readable.
    json redacted = root;
    if (redacted.contains("messages") && redacted["messages"].is_array()) {
        for (auto &msg : redacted["messages"]) {
            if (!msg.is_object() || !msg.contains("content")) {
                continue;
            }
            auto &content = msg["content"];
            if (content.is_array()) {
                for (auto &part : content) {
                    if (!part.is_object()) {
                        continue;
                    }
                    if (part.value("type", "") == "image_url" && part.contains("image_url") && part["image_url"].is_object()) {
                        if (part["image_url"].contains("url") && part["image_url"]["url"].is_string()) {
                            part["image_url"]["url"] = "<redacted>";
                        }
                    }
                }
            }
        }
    }
    return redacted;
}

static void log_ollama_chat_body(const oatpp::Object<ChatParams> &generation_params)
{
    if (!generation_params) {
        OATPP_LOGi("request", "/api/chat body: <null>");
        return;
    }
    json root;
    root["model"] = generation_params->model ? generation_params->model->c_str() : "";
    root["stream"] = generation_params->stream ? true : false;

    // Log messages with truncated content to see roles and whether any tool calls, etc.
    json msgs = json::array();
    if (generation_params->messages) {
        for (const auto &m : *generation_params->messages) {
            if (!m) continue;
            json jm;
            jm["role"] = m->role ? m->role->c_str() : "";
            jm["content"] = truncate_for_log(m->content ? m->content->c_str() : "", 256);
            msgs.push_back(std::move(jm));
        }
    }
    root["messages"] = std::move(msgs);

    // Only log image count, not raw base64
    size_t img_count = 0;
    if (generation_params->images) {
        for (const auto &img_b64 : *generation_params->images) {
            if (img_b64 && !img_b64->empty()) {
                img_count++;
            }
        }
    }
    root["image_count"] = img_count;

    OATPP_LOGi("request", "/api/chat parsed={}", truncate_for_log(root.dump(), 2048));
}

static void apply_openwebui_details_compat(const ModelInfo &model_data, const oatpp::Object<ModelInfoDetails> &details)
{
    if (!details) {
        return;
    }
    if (!webui_compat_enabled()) {
        return;
    }
    if (model_data.type != ModelType::VLM) {
        return;
    }

    // Spoof to a known vision family/format so OpenWebUI enables image pipeline.
    details->family = "llava";
    details->families = {};
    details->families->push_back("llava");
    details->format = "gguf";
}

// Apply user options to generator params (override defaults)
void apply_user_options(const oatpp::Object<ModelParameters> &options,
    hailort::genai::LLMGeneratorParams &generator_params)
{
    if (!options) {
        return;
    }

    if (options->temperature == 0.0F) {
        generator_params.set_do_sample(false);
    } else if (options->temperature != nullptr) {
        generator_params.set_do_sample(true);
        generator_params.set_temperature(options->temperature);
    }

    if (options->seed != nullptr && options->seed != -1) {
        generator_params.set_seed(options->seed);
    }

    if (options->top_k != nullptr) {
        generator_params.set_top_k(options->top_k);
    }
    if (options->top_p != nullptr) {
        generator_params.set_top_p(options->top_p);
    }
    if (options->frequency_penalty != nullptr) {
        generator_params.set_frequency_penalty(options->frequency_penalty);
    }

    if (options->num_predict != nullptr) {
        generator_params.set_max_generated_tokens(options->num_predict);
    }
}

MyController::MyController(const std::shared_ptr<SyncGenerationContext> &generation_context,
    const std::shared_ptr<ModelStore> &model_store, const std::shared_ptr<ResourceProvider> &resource_provider,
    const std::shared_ptr<oatpp::web::mime::ContentMappers> &apiContentMappers)
    : oatpp::web::server::api::ApiController(apiContentMappers), m_generation_context(generation_context),
      m_model_store(model_store), m_resource_provider(resource_provider)
{
}

std::optional<std::pair<ModelInfo, std::filesystem::path>> MyController::get_model_data(const std::string &model_name)
{
    const auto model_data_opt = m_model_store->get_model(model_name);
    if (!model_data_opt) {
        return std::nullopt;
    }
    const auto &model_data = *model_data_opt;
    const auto hef = m_resource_provider->get_resource(model_data.hef_resource);
    if (!fs::is_regular_file(hef)) {
        return std::nullopt;
    }

    return {{model_data, hef}};
}

std::optional<oatpp::Object<ModelInfoShort>> MyController::get_model_info(const std::string &model_name)
{
    const auto model_data_opt = get_model_data(model_name);
    if (!model_data_opt) {
        return std::nullopt;
    }
    const auto &model_data = model_data_opt->first;
    const auto &hef = model_data_opt->second;
    std::error_code error_code;
    const auto modified_at = fs::last_write_time(hef, error_code);
    if (error_code) {
        return std::nullopt;
    }
    const auto file_size = fs::file_size(hef, error_code);
    if (error_code) {
        return std::nullopt;
    }
    auto model_info = ModelInfoShort::createShared();
    model_info->name = model_name;
    model_info->model = model_name;
    model_info->size = file_size;
    model_info->modified_at = to_iso_8601(modified_at, "Z");
    if (!model_data.details.empty()) {
        model_info->details =
            m_contentMappers->getDefaultMapper()->readFromString<oatpp::Object<ModelInfoDetails>>(model_data.details);
    }
    apply_openwebui_details_compat(model_data, model_info->details);
    // Some clients (e.g. Open WebUI) only call /api/tags, so include vision capability here too.
    model_info->capabilities = {};
    if (model_data.type == ModelType::VLM) {
        model_info->capabilities->push_back("vision");
    }
    return model_info;
}

std::optional<std::chrono::seconds> MyController::convert_keep_alive(const oatpp::Int32 &keep_alive)
{
    if (!keep_alive) {
        return config::HAILO_OLLAMA_DEFAULT_KEEP_ALIVE; // default value for keep_alive is 5m
    } else if (*keep_alive < 0) {
        return std::nullopt;
    } else {
        return std::chrono::seconds(*keep_alive);
    }
}

std::shared_ptr<oat::OutgoingResponse> MyController::handle_completion(const ModelInfo &model_data,
    const std::vector<std::string> &prompt_json_strings, const oatpp::Object<ModelParameters> &options,
    const bool stream, const oatpp::Int32 &keep_alive, const std::string &model, const ReturnType return_type)
{
    using GenerationStatus = hailort::genai::LLMGeneratorCompletion::Status;

    const auto hef = m_resource_provider->get_resource(model_data.hef_resource);
    OATPP_LOGi("handle_completion", "Got model '{}', path '{}'", model, hef.string());

    // Load model first to get default generator params
    auto generator = m_generation_context->lock();
    generator->load_model(model_data.name, hef, convert_keep_alive(keep_alive));

    // Get EOS token from model to clamp it from responses
    std::string eos_token = generator->get_generation_recovery_sequence();

    // Get default generator params from LLM API
    auto generator_params = generator->create_generator_params();

    // Apply user options (overriding defaults)
    apply_user_options(options, generator_params);

    Generation generation{
        model_data.name,
        hef,
        prompt_json_strings,
        std::optional<hailort::genai::LLMGeneratorParams>(std::move(generator_params)),
        {}, // options handled via generator_params for LLM
        convert_keep_alive(keep_alive)
    };

    auto generator_completion = generator->generate_one(std::move(generation));
    if (!stream) {
        const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        uint32_t token_count = 0;
        std::string response_text = "";
        while (generator_completion.generation_status() == GenerationStatus::GENERATING) {
            response_text += generator_completion.read().expect("read failed!");
            token_count++;
        }

        // Strip EOS token from response if present
        response_text = strip_eos_suffix(response_text, eos_token);

        const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        const auto status = generator_completion.generation_status();
        const char *stop_reason = (status == GenerationStatus::LOGICAL_END_OF_GENERATION)
                                      ? GenerationContext::GenerationFinishedReason::STOP
                                      : GenerationContext::GenerationFinishedReason::LENGTH;

        // Append assistant response to conversation history for context continuation
        generator->append_assistant_message(response_text);

        generator.reset(); // Unlock before creating response

        if (return_type == ReturnType::COMPLETION) {
            auto result = CreateChatCompletionResponse::createShared();
            result->id = "chatcmpl-" + std::to_string(std::rand());
            result->object = "chat.completion";
            result->created = std::chrono::system_clock::now().time_since_epoch().count();

            result->model = model;
            result->choices = {};

            auto message = ChatCompletionMessage::createShared();
            message->role = "assistant";
            message->content = response_text;

            auto choice = ChatChoice::createShared();
            choice->index = oatpp::Int64(static_cast<int64_t>(0));
            choice->finish_reason = std::move(stop_reason);
            choice->message = message;

            result->choices->push_back(choice);
            return createDtoResponse(Status::CODE_200, result);
        }
        auto result = GenerationResponseFinal::createShared();
        result->model = model;
        result->created_at = get_current_time_formatted();
        if (return_type == ReturnType::MESSAGE) {
            result->message = ChatMessage::createShared();
            result->message->role = "assistant";
            result->message->content = response_text;
        } else {
            result->response = response_text;
        }

        result->done = true;
        result->done_reason = std::move(stop_reason);
        const auto total_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        result->total_duration = total_time_ns;
        result->eval_count = token_count;

        return createDtoResponse(Status::CODE_200, result);
    }
    auto body = std::make_shared<oat::OutgoingStreamingBody>(std::make_shared<LLMGenerationReadCallback>(model,
        m_contentMappers->getDefaultMapper(), std::move(generator), std::move(generator_completion),
        return_type == ReturnType::MESSAGE, eos_token));

    auto outgoing_response = OutgoingResponse::createShared(Status::CODE_200, body);
    outgoing_response->putHeader("Content-Type", "application/x-ndjson");
    return outgoing_response;
}

std::shared_ptr<oat::OutgoingResponse> MyController::handle_vlm_completion(const ModelInfo &model_data,
    ExtractedImages &&images, const oatpp::Object<ModelParameters> &options, const bool stream,
    const oatpp::Int32 &keep_alive, const std::string &model, const ReturnType return_type)
{
    using GenerationStatus = hailort::genai::LLMGeneratorCompletion::Status;

    const auto hef = m_resource_provider->get_resource(model_data.hef_resource);
    OATPP_LOGi("handle_vlm_completion", "Got model '{}', path '{}'", model, hef.string());

    auto generator = m_generation_context->lock();
    generator->load_model(model_data.name, hef, convert_keep_alive(keep_alive), ModelType::VLM);

    Generation generation{
        model_data.name,
        hef,
        {}, // prompt_json_strings unused for VLM
        std::nullopt, // VLM does not use LLMGeneratorParams here (created inside generate_one_vlm)
        {},
        convert_keep_alive(keep_alive),
        ModelType::VLM,
        std::move(images.frames),
        std::move(images.messages_json),
    };

    // Apply user options to VLM generation params (LLMGeneratorParams is shared).
    if (options) {
        if (options->temperature != nullptr) generation.options.temperature = options->temperature;
        if (options->seed != nullptr) generation.options.seed = options->seed;
        if (options->top_k != nullptr) generation.options.top_k = options->top_k;
        if (options->top_p != nullptr) generation.options.top_p = options->top_p;
        if (options->frequency_penalty != nullptr) generation.options.frequency_penalty = options->frequency_penalty;
        if (options->num_predict != nullptr) generation.options.num_predict = options->num_predict;
    }
    // Protect against "never-ending" streaming in UI clients when max tokens is omitted.
    // Only apply this safeguard for streaming responses; non-streaming should rely on the model's stop tokens.
    if (stream && !generation.options.num_predict.has_value()) {
        generation.options.num_predict = 256;
    }

    auto generator_completion = generator->generate_one_vlm(std::move(generation));

    if (!stream) {
        const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        uint32_t token_count = 0;
        std::string response_text;
        while (generator_completion.generation_status() == GenerationStatus::GENERATING) {
            response_text += generator_completion.read().expect("read failed!");
            token_count++;
        }

        const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        const auto status = generator_completion.generation_status();
        const char *stop_reason = (status == GenerationStatus::LOGICAL_END_OF_GENERATION)
                                      ? GenerationContext::GenerationFinishedReason::STOP
                                      : GenerationContext::GenerationFinishedReason::LENGTH;

        generator.reset(); // Unlock before creating response

        if (return_type == ReturnType::COMPLETION) {
            auto result = CreateChatCompletionResponse::createShared();
            result->id = "chatcmpl-" + std::to_string(std::rand());
            result->object = "chat.completion";
            result->created = std::chrono::system_clock::now().time_since_epoch().count();
            result->model = model;
            result->choices = {};

            auto message = ChatCompletionMessage::createShared();
            message->role = "assistant";
            message->content = response_text;

            auto choice = ChatChoice::createShared();
            choice->index = oatpp::Int64(static_cast<int64_t>(0));
            choice->finish_reason = std::move(stop_reason);
            choice->message = message;
            result->choices->push_back(choice);
            return createDtoResponse(Status::CODE_200, result);
        }

        auto result = GenerationResponseFinal::createShared();
        result->model = model;
        result->created_at = get_current_time_formatted();
        if (return_type == ReturnType::MESSAGE) {
            result->message = ChatMessage::createShared();
            result->message->role = "assistant";
            result->message->content = response_text;
        } else {
            result->response = response_text;
        }
        result->done = true;
        result->done_reason = std::move(stop_reason);
        result->total_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        result->eval_count = token_count;
        return createDtoResponse(Status::CODE_200, result);
    }

    auto body = std::make_shared<oat::OutgoingStreamingBody>(std::make_shared<LLMGenerationReadCallback>(model,
        m_contentMappers->getDefaultMapper(), std::move(generator), std::move(generator_completion),
        return_type == ReturnType::MESSAGE, "" /* eos_token */));

    auto outgoing_response = OutgoingResponse::createShared(Status::CODE_200, body);
    outgoing_response->putHeader("Content-Type", "application/x-ndjson");
    return outgoing_response;
}

std::shared_ptr<oat::OutgoingResponse> MyController::handle_load_unload(const std::string &model_name,
    const ModelInfo &model_data, const oatpp::Object<ModelParameters> &options, const oatpp::Int32 &keep_alive,
    const bool return_as_message)
{
    (void)options;
    // keep alive is 0 -> should unload the model
    auto result = GenerationResponseFinal::createShared();
    auto generator = m_generation_context->lock();
    if (keep_alive && *keep_alive == 0) {
        generator->reset();

        result->done_reason = "unload";
    } else {
        // Model load
        const auto hef = m_resource_provider->get_resource(model_data.hef_resource);
        generator->load_model(model_name, hef, convert_keep_alive(keep_alive));
        result->done_reason = "load";
    }
    result->model = model_name;
    result->created_at = get_current_time_formatted();
    if (return_as_message) {
        result->message = ChatMessage::createShared();
        result->message->role = "assistant";
        result->message->content = "";
    } else {
        result->response = "";
    }
    result->done = true;
    return createDtoResponse(Status::CODE_200, result);
}

std::shared_ptr<oat::OutgoingResponse> MyController::root()
{
    return ResponseFactory::createResponse(Status::CODE_200, "hailo-ollama is running");
}

std::shared_ptr<oat::OutgoingResponse> MyController::health()
{
    return ResponseFactory::createResponse(Status::CODE_200, "ok");
}

std::shared_ptr<oat::OutgoingResponse> MyController::version()
{
    auto result = VersionResponse::createShared();
    // tools might rely on this -> return a version similar to original Ollama
    result->version = "0.5.1";
    return createDtoResponse(Status::CODE_200, result);
}

std::shared_ptr<oat::OutgoingResponse> MyController::list_models()
{
    const auto model_names = m_model_store->get_model_names();
    OATPP_LOGi("list_models", "got {} models in store", model_names.size());
    auto result = TagsResponse::createShared();
    result->models = {};
    for (const auto &model_name : model_names) {
        OATPP_LOGi("list_models", "model: {}", model_name);
        const auto model_info = get_model_info(model_name);
        if (!model_info) {
            continue;
        }
        result->models->push_back(*model_info);
    }
    return createDtoResponse(Status::CODE_200, result);
}

std::shared_ptr<oat::OutgoingResponse> MyController::v1_models()
{
    json resp;
    resp["object"] = "list";
    resp["data"] = json::array();
    for (const auto &name : m_model_store->get_model_names()) {
        resp["data"].push_back({
            {"id", name},
            {"object", "model"},
            {"created", 0},
            {"owned_by", "openai"}
        });
    }
    auto r = createResponse(Status::CODE_200, resp.dump());
    r->putHeader("Content-Type", "application/json");
    return r;
}

std::shared_ptr<oat::OutgoingResponse> MyController::v1_model_by_id(const oatpp::String &id)
{
    if (!id) {
        return createResponse(Status::CODE_404, "");
    }
    const auto model_data_opt = get_model_data(id->c_str());
    if (!model_data_opt) {
        return createResponse(Status::CODE_404, "");
    }
    json obj;
    obj["id"] = id->c_str();
    obj["object"] = "model";
    obj["created"] = 0;
    obj["owned_by"] = "openai";
    auto r = createResponse(Status::CODE_200, obj.dump());
    r->putHeader("Content-Type", "application/json");
    return r;
}

std::shared_ptr<oat::OutgoingResponse> MyController::list_all_models()
{
    const auto model_names = m_model_store->get_model_names();
    OATPP_LOGi("list_models", "got {} models in store", model_names.size());
    auto result = ListAllResponse::createShared();
    result->models = {};
    for (const auto &name : model_names) {
        result->models->push_back(name);
    }

    return createDtoResponse(Status::CODE_200, result);
}

std::shared_ptr<oat::OutgoingResponse> MyController::list_running_models()
{
    auto generator = m_generation_context->lock();
    const auto model_name = generator->get_model_name();
    const auto expiration = generator->get_expiration();
    generator.reset(); // unlock
    auto result = TagsResponse::createShared();
    result->models = {};
    const auto model_info_opt = get_model_info(model_name);
    if (model_info_opt) {
        const auto &model_info = *model_info_opt;
        model_info->expires_at = to_iso_8601(expiration, "Z");
        result->models->push_back(model_info);
    }

    return createDtoResponse(Status::CODE_200, result);
}

std::shared_ptr<oat::OutgoingResponse> MyController::show(const oatpp::Object<ShowParams> &show_params)
{
    const auto model_data_opt = get_model_data(show_params->model);
    if (!model_data_opt) {
        auto error_result = ErrorResponse::createShared();
        error_result->error = "model '" + show_params->model + "' not found";
        return createDtoResponse(Status::CODE_404, error_result);
    }
    const auto &model_data = model_data_opt->first;
    auto result = ShowResponse::createShared();
    result->license = model_data.license;
    result->modelfile = "";
    result->parameters = ""; // Stop tokens are now managed by HailoRT

    // Get HEF path and file modification time
    const auto &hef_path = model_data_opt->second;
    std::error_code error_code;
    const auto modified_at = fs::last_write_time(hef_path, error_code);
    if (!error_code) {
        result->modified_at = to_iso_8601(modified_at, "Z");
    }

    if (!model_data.details.empty()) {
        result->details =
            m_contentMappers->getDefaultMapper()->readFromString<oatpp::Object<ModelInfoDetails>>(model_data.details);
    }
    apply_openwebui_details_compat(model_data, result->details);
    // Advertise vision capability so clients (e.g. Open WebUI) know this model accepts images.
    result->capabilities = {};
    if (model_data.type == ModelType::VLM) {
        result->capabilities->push_back("vision");
    }
    result->model_info = "";
    return createDtoResponse(Status::CODE_200, result);
}

std::shared_ptr<oat::OutgoingResponse> MyController::pull_model(const oatpp::Object<PullParams> &pull_params)
{
    auto model_data = m_model_store->get_model(pull_params->model);
    if (!model_data) {
        auto error_result = ErrorResponse::createShared();
        error_result->error = "model '" + pull_params->model + "' not found";
        return createDtoResponse(Status::CODE_404, error_result);
    }

    if (!pull_params->stream) {
        try {
            m_resource_provider->pull_resource(model_data->hef_resource);
        } catch (const std::exception &e) {
            auto error_result = ErrorResponse::createShared();
            error_result->error = e.what();
            return createDtoResponse(Status::CODE_500, error_result);
        }
        auto result = PullResponse::createShared();
        result->status = "success";

        return createDtoResponse(Status::CODE_200, result);
    }

    auto queue = std::make_shared<PullReadCallback::EventQueue>();
    std::thread pull_thread([this, &model_data, queue, hef_resource = model_data->hef_resource]() {
        m_resource_provider->pull_resource(hef_resource, queue);
    });
    auto body =
        std::make_shared<oat::OutgoingStreamingBody>(std::make_shared<PullReadCallback>(m_contentMappers
                                                                                            ->getDefaultMapper(),
            queue, std::move(pull_thread)));

    auto outgoing_response = OutgoingResponse::createShared(Status::CODE_200, body);
    outgoing_response->putHeader("Content-Type", "application/x-ndjson");
    return outgoing_response;
}

std::shared_ptr<oat::OutgoingResponse> MyController::delete_model(const oatpp::Object<DeleteParams> &delete_params)
{
    auto model_data = m_model_store->get_model(delete_params->model);
    if (!model_data) {
        auto error_result = DeleteErrorResponse::createShared();
        error_result->code = "not_found";
        error_result->error = "model not found";
        return createDtoResponse(Status::CODE_404, error_result);
    }
    const auto hef = m_resource_provider->get_resource(model_data->hef_resource);
    std::error_code error_code;
    const auto removed = fs::remove(hef, error_code);
    if (error_code || !removed) {
        auto error_result = DeleteErrorResponse::createShared();
        error_result->code = "not_found";
        error_result->error = "model not found";
        return createDtoResponse(Status::CODE_404, error_result);
    }

    return createResponse(Status::CODE_200);
}

std::shared_ptr<oat::OutgoingResponse> MyController::generate(const oatpp::Object<GenerationParams> &generation_params)
{
    const auto &model = generation_params->model;

    if (debug_requests_enabled()) {
        const size_t img_count = (generation_params->images) ? generation_params->images->size() : 0;
        OATPP_LOGi("request", "/api/generate model='{}' stream={} images={}",
            std::string(model), generation_params->stream ? "true" : "false", img_count);
    }

    const auto model_data_opt = get_model_data(model);
    if (!model_data_opt) {
        auto error_result = ErrorResponse::createShared();
        error_result->error = "model '" + model + "' not found";
        return createDtoResponse(Status::CODE_404, error_result);
    }
    const auto &model_data = model_data_opt->first;

    if (!generation_params->prompt) {
        return handle_load_unload(model, model_data, generation_params->options, generation_params->keep_alive, false);
    }

    const auto &prompt = generation_params->prompt;
    const auto stream = generation_params->stream;

    if (model_data.type == ModelType::VLM) {
        const uint32_t target_w = model_data.frame_width != 0 ? model_data.frame_width : 336;
        const uint32_t target_h = model_data.frame_height != 0 ? model_data.frame_height : 336;

        ExtractedImages extracted;
        try {
            if (generation_params->images) {
                for (const auto &img_b64 : *generation_params->images) {
                    if (!img_b64) {
                        continue;
                    }
                    const auto raw = base64_decode(img_b64->c_str());
                    const auto frame = decode_and_resize_image(raw, target_w, target_h);
                    extracted.frames.push_back(frame.data);
                }
            }
        } catch (const std::exception &e) {
            auto error_result = ErrorResponse::createShared();
            error_result->error = std::string("invalid image: ") + e.what();
            return createDtoResponse(Status::CODE_400, error_result);
        }

        json msg;
        msg["role"] = "user";
        if (!extracted.frames.empty()) {
            json content_arr = json::array();
            for (size_t i = 0; i < extracted.frames.size(); ++i) {
                content_arr.push_back({{"type", "image"}});
            }
            content_arr.push_back({{"type", "text"}, {"text", std::string(prompt)}});
            msg["content"] = content_arr;
        } else {
            // Text-only request for a VLM model: send simple string content and no frames.
            msg["content"] = std::string(prompt);
        }
        extracted.messages_json.push_back(msg.dump());

        return handle_vlm_completion(model_data, std::move(extracted), generation_params->options, stream,
            generation_params->keep_alive, model, ReturnType::RESPONSE);
    }

    // Create structured prompt as JSON string using raw string literal
    std::vector<std::string> prompt_json_strings = {R"({"role": "user", "content": ")" + std::string(prompt) + R"("})"};

    return handle_completion(model_data, prompt_json_strings, generation_params->options, stream,
        generation_params->keep_alive, model, ReturnType::RESPONSE);
}

std::shared_ptr<oat::OutgoingResponse> MyController::chat(const oatpp::Object<ChatParams> &generation_params)
{
    const auto &model = generation_params->model;
    if (debug_requests_enabled()) {
        const size_t msg_count = (generation_params->messages) ? generation_params->messages->size() : 0;
        const size_t img_count = (generation_params->images) ? generation_params->images->size() : 0;
        OATPP_LOGi("request", "/api/chat model='{}' stream={} messages={} images={}",
            std::string(model), generation_params->stream ? "true" : "false", msg_count, img_count);
        log_ollama_chat_body(generation_params);
    }
    const auto model_data_opt = get_model_data(model);
    if (!model_data_opt) {
        auto error_result = ErrorResponse::createShared();
        error_result->error = "model '" + model + "' not found";
        return createDtoResponse(Status::CODE_404, error_result);
    }
    const auto &model_data = model_data_opt->first;

    if (!generation_params->messages) {
        return handle_load_unload(model, model_data, generation_params->options, generation_params->keep_alive, true);
    }

    const auto stream = generation_params->stream;

    if (model_data.type == ModelType::VLM) {
        const uint32_t target_w = model_data.frame_width != 0 ? model_data.frame_width : 336;
        const uint32_t target_h = model_data.frame_height != 0 ? model_data.frame_height : 336;

        ExtractedImages extracted;
        try {
            if (generation_params->images) {
                for (const auto &img_b64 : *generation_params->images) {
                    if (!img_b64) {
                        continue;
                    }
                    const auto raw = base64_decode(img_b64->c_str());
                    const auto frame = decode_and_resize_image(raw, target_w, target_h);
                    extracted.frames.push_back(frame.data);
                }
            }
        } catch (const std::exception &e) {
            auto error_result = ErrorResponse::createShared();
            error_result->error = std::string("invalid image: ") + e.what();
            return createDtoResponse(Status::CODE_400, error_result);
        }

        bool image_inserted = false;
        for (const auto &message : *generation_params->messages) {
            json vlm_msg;
            vlm_msg["role"] = message->role ? message->role->c_str() : "user";
            if (!image_inserted && !extracted.frames.empty() && message->role && message->role == "user") {
                json content_arr = json::array();
                for (size_t i = 0; i < extracted.frames.size(); ++i) {
                    content_arr.push_back({{"type", "image"}});
                }
                content_arr.push_back({{"type", "text"}, {"text", message->content ? message->content->c_str() : ""}});
                vlm_msg["content"] = content_arr;
                image_inserted = true;
            } else {
                vlm_msg["content"] = message->content ? message->content->c_str() : "";
            }
            extracted.messages_json.push_back(vlm_msg.dump());
        }

        return handle_vlm_completion(model_data, std::move(extracted), generation_params->options, stream,
            generation_params->keep_alive, model, ReturnType::MESSAGE);
    }

    // Convert messages to JSON strings for structured prompts
    std::vector<std::string> prompt_json_strings;
    for (const auto &message : *generation_params->messages) {
        prompt_json_strings.push_back(R"({"role": ")" + message->role + R"(", "content": ")" +
                                      escape_json_quotes(message->content) + R"("})");
    }

    return handle_completion(model_data, prompt_json_strings, generation_params->options, stream,
        generation_params->keep_alive, model, ReturnType::MESSAGE);
}

bool MyController::openai_messages_contain_image_url(const nlohmann::ordered_json &messages)
{
    if (!messages.is_array()) {
        return false;
    }
    for (const auto &msg : messages) {
        if (!msg.contains("content") || !msg["content"].is_array()) {
            continue;
        }
        for (const auto &part : msg["content"]) {
            if (part.value("type", "") == "image_url") {
                return true;
            }
        }
    }
    return false;
}

MyController::ExtractedImages MyController::extract_openai_vision_messages(
    const nlohmann::ordered_json &messages, uint32_t target_w, uint32_t target_h)
{
    ExtractedImages result;
    if (!messages.is_array()) {
        return result;
    }
    for (const auto &msg : messages) {
        const auto role = msg.value("role", std::string("user"));
        json vlm_msg;
        vlm_msg["role"] = role;

        if (msg.contains("content") && msg["content"].is_array()) {
            json content_arr = json::array();
            for (const auto &part : msg["content"]) {
                if (part.value("type", "") == "image_url") {
                    const auto &image_url = part["image_url"];
                    const std::string url = image_url.is_object() ? image_url.value("url", std::string()) : "";
                    const auto raw = base64_decode(url);
                    auto frame = decode_and_resize_image(raw, target_w, target_h);
                    result.frames.push_back(std::move(frame.data));
                    content_arr.push_back(json{{"type", "image"}});
                } else if (part.value("type", "") == "text") {
                    content_arr.push_back(json{{"type", "text"}, {"text", part.value("text", std::string())}});
                }
            }
            vlm_msg["content"] = std::move(content_arr);
        } else if (msg.contains("content") && msg["content"].is_string()) {
            vlm_msg["content"] = msg["content"].get<std::string>();
        } else {
            vlm_msg["content"] = "";
        }
        result.messages_json.push_back(vlm_msg.dump());
    }
    return result;
}

std::vector<std::string> MyController::openai_messages_to_prompt_json_strings(const nlohmann::ordered_json &messages)
{
    std::vector<std::string> out;
    if (!messages.is_array()) {
        return out;
    }
    for (const auto &msg : messages) {
        const std::string role = msg.value("role", std::string("user"));
        std::string flat;
        if (msg.contains("content") && !msg["content"].is_null()) {
            const auto &c = msg["content"];
            if (c.is_string()) {
                flat = c.get<std::string>();
            } else if (c.is_array()) {
                for (const auto &part : c) {
                    if (part.value("type", "") == "text") {
                        flat += part.value("text", std::string());
                    }
                }
            }
        }
        out.push_back(R"({"role": ")" + role + R"(", "content": ")" + escape_json_quotes(flat) + R"("})");
    }
    return out;
}

std::shared_ptr<oat::OutgoingResponse> MyController::chat_completions(const oatpp::String &body)
{
    if (debug_requests_enabled()) {
        OATPP_LOGi("request", "/v1/chat/completions raw_body={}", truncate_for_log(body ? body->c_str() : "", 2048));
    }

    if (!body) {
        auto error_result = ErrorResponse::createShared();
        error_result->error = "empty body";
        return createDtoResponse(Status::CODE_400, error_result);
    }

    json root;
    try {
        root = json::parse(body->c_str());
    } catch (const std::exception &e) {
        auto error_result = ErrorResponse::createShared();
        error_result->error = std::string("invalid json: ") + e.what();
        return createDtoResponse(Status::CODE_400, error_result);
    }

    if (debug_requests_enabled()) {
        const auto redacted = redact_images_in_openai_body(root);
        OATPP_LOGi("request", "/v1/chat/completions parsed={}", truncate_for_log(redacted.dump(), 4096));
    }

    if (!root.contains("messages") || !root["messages"].is_array()) {
        auto error_result = ErrorResponse::createShared();
        error_result->error = "messages field is required";
        return createDtoResponse(Status::CODE_400, error_result);
    }
    const auto &messages = root["messages"];

    const int n = root.value("n", 1);
    if (n != 1) {
        auto error_result = ErrorResponse::createShared();
        error_result->error = "only n == 1 is supported";
        return createDtoResponse(Status::CODE_400, error_result);
    }

    const bool stream = root.value("stream", false);

    if (!root.contains("model") || !root["model"].is_string()) {
        auto error_result = ErrorResponse::createShared();
        error_result->error = "model field is required";
        return createDtoResponse(Status::CODE_400, error_result);
    }
    const std::string model = root["model"].get<std::string>();

    const auto model_data_opt = get_model_data(model);
    if (!model_data_opt) {
        auto error_result = ErrorResponse::createShared();
        error_result->error = "model '" + model + "' not found";
        return createDtoResponse(Status::CODE_404, error_result);
    }
    const auto &model_data = model_data_opt->first;

    const uint32_t target_w = model_data.frame_width != 0 ? model_data.frame_width : 336;
    const uint32_t target_h = model_data.frame_height != 0 ? model_data.frame_height : 336;

    const bool has_images = openai_messages_contain_image_url(messages);
    if (has_images && model_data.type != ModelType::VLM) {
        auto error_result = ErrorResponse::createShared();
        error_result->error = "vision (image_url) requires a model with type \"vlm\" in its manifest";
        return createDtoResponse(Status::CODE_400, error_result);
    }

    if (model_data.type == ModelType::VLM && has_images) {
        try {
            auto extracted = extract_openai_vision_messages(messages, target_w, target_h);
            if (extracted.frames.empty()) {
                auto error_result = ErrorResponse::createShared();
                error_result->error =
                    "no decodable images found in messages (expected image_url.url as data URL or raw base64)";
                return createDtoResponse(Status::CODE_400, error_result);
            }

            auto model_options = ModelParameters::createShared();
            if (root.contains("temperature") && !root["temperature"].is_null()) {
                model_options->temperature = root["temperature"].get<float>();
            }
            if (root.contains("seed") && !root["seed"].is_null()) {
                model_options->seed = root["seed"].get<int32_t>();
            }
            if (root.contains("top_p") && !root["top_p"].is_null()) {
                model_options->top_p = root["top_p"].get<float>();
            }
            if (root.contains("frequency_penalty") && !root["frequency_penalty"].is_null()) {
                model_options->frequency_penalty = root["frequency_penalty"].get<float>();
            }
            if (root.contains("max_tokens") && !root["max_tokens"].is_null()) {
                model_options->num_predict = root["max_tokens"].get<uint32_t>();
            } else if (root.contains("max_completion_tokens") && !root["max_completion_tokens"].is_null()) {
                model_options->num_predict = root["max_completion_tokens"].get<uint32_t>();
            }

            if (!stream) {
                return handle_vlm_completion(model_data, std::move(extracted), model_options, stream, oatpp::Int32(nullptr), model,
                    ReturnType::COMPLETION);
            }

            // Streaming (SSE): run VLM and stream deltas
            auto generator = m_generation_context->lock();
            generator->load_model(model_data.name, model_data_opt->second, convert_keep_alive(oatpp::Int32(nullptr)), ModelType::VLM);

            Generation generation{
                model_data.name,
                model_data_opt->second,
                {}, // prompt_json_strings unused for VLM
                std::nullopt,
                {},
                convert_keep_alive(oatpp::Int32(nullptr)),
                ModelType::VLM,
                std::move(extracted.frames),
                std::move(extracted.messages_json),
            };
            if (model_options) {
                if (model_options->temperature != nullptr) generation.options.temperature = model_options->temperature;
                if (model_options->seed != nullptr) generation.options.seed = model_options->seed;
                if (model_options->top_k != nullptr) generation.options.top_k = model_options->top_k;
                if (model_options->top_p != nullptr) generation.options.top_p = model_options->top_p;
                if (model_options->frequency_penalty != nullptr) generation.options.frequency_penalty = model_options->frequency_penalty;
                if (model_options->num_predict != nullptr) generation.options.num_predict = model_options->num_predict;
            }
            // OpenAI clients often omit max_tokens; use a larger default than Ollama to avoid "early truncation".
            if (!generation.options.num_predict.has_value()) {
                generation.options.num_predict = 1024;
            }

            auto completion = generator->generate_one_vlm(std::move(generation));
            generator.reset();

            const std::string id = "chatcmpl-" + std::to_string(std::rand());
            auto body_stream = std::make_shared<oat::OutgoingStreamingBody>(
                std::make_shared<OpenAIStreamingReadCallback>(id, model, std::move(completion)));
            auto resp = OutgoingResponse::createShared(Status::CODE_200, body_stream);
            resp->putHeader("Content-Type", "text/event-stream");
            resp->putHeader("Cache-Control", "no-cache");
            return resp;
        } catch (const std::exception &e) {
            auto error_result = ErrorResponse::createShared();
            error_result->error = std::string("invalid image or vision payload: ") + e.what();
            return createDtoResponse(Status::CODE_400, error_result);
        }
    }

    // Text-only requests to a VLM model must still go through the VLM path (frames can be empty).
    // Otherwise we may end up invoking LLM APIs on a VLM HEF, which can produce corrupted output.
    if (model_data.type == ModelType::VLM && !has_images) {
        ExtractedImages extracted;
        for (const auto &msg : messages) {
            const std::string role = msg.value("role", std::string("user"));
            std::string flat;
            if (msg.contains("content") && !msg["content"].is_null()) {
                const auto &c = msg["content"];
                if (c.is_string()) {
                    flat = c.get<std::string>();
                } else if (c.is_array()) {
                    for (const auto &part : c) {
                        if (part.value("type", "") == "text") {
                            flat += part.value("text", std::string());
                        }
                    }
                }
            }

            json vlm_msg;
            vlm_msg["role"] = role;
            vlm_msg["content"] = flat;
            extracted.messages_json.push_back(vlm_msg.dump());
        }

        auto model_options = ModelParameters::createShared();
        if (root.contains("temperature") && !root["temperature"].is_null()) {
            model_options->temperature = root["temperature"].get<float>();
        }
        if (root.contains("seed") && !root["seed"].is_null()) {
            model_options->seed = root["seed"].get<int32_t>();
        }
        if (root.contains("top_p") && !root["top_p"].is_null()) {
            model_options->top_p = root["top_p"].get<float>();
        }
        if (root.contains("frequency_penalty") && !root["frequency_penalty"].is_null()) {
            model_options->frequency_penalty = root["frequency_penalty"].get<float>();
        }
        if (root.contains("max_tokens") && !root["max_tokens"].is_null()) {
            model_options->num_predict = root["max_tokens"].get<uint32_t>();
        } else if (root.contains("max_completion_tokens") && !root["max_completion_tokens"].is_null()) {
            model_options->num_predict = root["max_completion_tokens"].get<uint32_t>();
        }

        if (!stream) {
            return handle_vlm_completion(model_data, std::move(extracted), model_options, stream, oatpp::Int32(nullptr),
                model, ReturnType::COMPLETION);
        }

        // Streaming (SSE) VLM text-only path
        auto generator = m_generation_context->lock();
        generator->load_model(model_data.name, model_data_opt->second, convert_keep_alive(oatpp::Int32(nullptr)), ModelType::VLM);

        Generation generation{
            model_data.name,
            model_data_opt->second,
            {}, // prompt_json_strings unused for VLM
            std::nullopt,
            {},
            convert_keep_alive(oatpp::Int32(nullptr)),
            ModelType::VLM,
            {}, // no frames
            std::move(extracted.messages_json),
        };
        if (model_options) {
            if (model_options->temperature != nullptr) generation.options.temperature = model_options->temperature;
            if (model_options->seed != nullptr) generation.options.seed = model_options->seed;
            if (model_options->top_k != nullptr) generation.options.top_k = model_options->top_k;
            if (model_options->top_p != nullptr) generation.options.top_p = model_options->top_p;
            if (model_options->frequency_penalty != nullptr) generation.options.frequency_penalty = model_options->frequency_penalty;
            if (model_options->num_predict != nullptr) generation.options.num_predict = model_options->num_predict;
        }
        // OpenAI clients often omit max_tokens; use a larger default than Ollama to avoid "early truncation".
        if (!generation.options.num_predict.has_value()) {
            generation.options.num_predict = 1024;
        }

        auto completion = generator->generate_one_vlm(std::move(generation));
        generator.reset();

        const std::string id = "chatcmpl-" + std::to_string(std::rand());
        auto body_stream = std::make_shared<oat::OutgoingStreamingBody>(
            std::make_shared<OpenAIStreamingReadCallback>(id, model, std::move(completion)));
        auto resp = OutgoingResponse::createShared(Status::CODE_200, body_stream);
        resp->putHeader("Content-Type", "text/event-stream");
        resp->putHeader("Cache-Control", "no-cache");
        return resp;
    }

    const std::vector<std::string> prompt_json_strings = openai_messages_to_prompt_json_strings(messages);
    if (prompt_json_strings.empty()) {
        auto error_result = ErrorResponse::createShared();
        error_result->error = "messages array is empty or invalid";
        return createDtoResponse(Status::CODE_400, error_result);
    }

    auto model_options = ModelParameters::createShared();
    if (root.contains("temperature") && !root["temperature"].is_null()) {
        model_options->temperature = root["temperature"].get<float>();
    }
    if (root.contains("seed") && !root["seed"].is_null()) {
        model_options->seed = root["seed"].get<int32_t>();
    }
    if (root.contains("top_p") && !root["top_p"].is_null()) {
        model_options->top_p = root["top_p"].get<float>();
    }
    if (root.contains("frequency_penalty") && !root["frequency_penalty"].is_null()) {
        model_options->frequency_penalty = root["frequency_penalty"].get<float>();
    }
    if (root.contains("max_tokens") && !root["max_tokens"].is_null()) {
        model_options->num_predict = root["max_tokens"].get<uint32_t>();
    } else if (root.contains("max_completion_tokens") && !root["max_completion_tokens"].is_null()) {
        model_options->num_predict = root["max_completion_tokens"].get<uint32_t>();
    }

    if (!stream) {
        return handle_completion(model_data, prompt_json_strings, model_options, stream, oatpp::Int32(nullptr), model,
            ReturnType::COMPLETION);
    }

    // Streaming (SSE) LLM path
    auto generator = m_generation_context->lock();
    generator->load_model(model_data.name, model_data_opt->second, convert_keep_alive(oatpp::Int32(nullptr)), ModelType::LLM);
    auto gen_params = generator->create_generator_params();
    apply_user_options(model_options, gen_params);
    Generation generation{
        model_data.name,
        model_data_opt->second,
        prompt_json_strings,
        std::optional<hailort::genai::LLMGeneratorParams>(std::move(gen_params)),
        {},
        convert_keep_alive(oatpp::Int32(nullptr))
    };
    auto completion = generator->generate_one(std::move(generation));
    generator.reset();

    const std::string id = "chatcmpl-" + std::to_string(std::rand());
    auto body_stream = std::make_shared<oat::OutgoingStreamingBody>(
        std::make_shared<OpenAIStreamingReadCallback>(id, model, std::move(completion)));
    auto resp = OutgoingResponse::createShared(Status::CODE_200, body_stream);
    resp->putHeader("Content-Type", "text/event-stream");
    resp->putHeader("Cache-Control", "no-cache");
    return resp;
}

} // namespace hailo_ollama
