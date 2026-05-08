/**
 * Copyright (c) 2019-2026 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
#include "utils/base64.hpp"

#include <stdexcept>
#include <string_view>

#include <openssl/bio.h>
#include <openssl/evp.h>

namespace hailo_ollama
{

std::vector<uint8_t> base64_decode(const std::string &input)
{
    // Strip data URI prefix (e.g. "data:image/jpeg;base64,")
    std::string_view data(input);
    const auto comma_pos = data.find(',');
    if (comma_pos != std::string_view::npos) {
        const auto prefix = data.substr(0, comma_pos);
        if (prefix.find("base64") != std::string_view::npos) {
            data = data.substr(comma_pos + 1);
        }
    }

    if (data.empty()) {
        throw std::runtime_error("base64_decode: empty input");
    }

    std::vector<uint8_t> result(data.size() * 3 / 4 + 4);

    BIO *b64 = BIO_new(BIO_f_base64());
    if (!b64) {
        throw std::runtime_error("base64_decode: BIO_new failed");
    }
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);

    BIO *bmem = BIO_new_mem_buf(data.data(), static_cast<int>(data.size()));
    if (!bmem) {
        BIO_free(b64);
        throw std::runtime_error("base64_decode: BIO_new_mem_buf failed");
    }
    bmem = BIO_push(b64, bmem);

    const int decoded_len = BIO_read(bmem, result.data(), static_cast<int>(result.size()));
    BIO_free_all(bmem);

    if (decoded_len < 0) {
        throw std::runtime_error("base64_decode: BIO_read failed");
    }

    result.resize(static_cast<size_t>(decoded_len));
    return result;
}

} // namespace hailo_ollama

