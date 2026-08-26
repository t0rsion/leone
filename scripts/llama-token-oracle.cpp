#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

static void leone_oracle_log_logits(llama_context * context, llama_token sampled) {
    if (std::getenv("LEONE_LLAMA_LOGITS") == nullptr) {
        return;
    }
    const llama_model * model = llama_get_model(context);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t count = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(context, -1);
    std::vector<std::pair<float, llama_token>> ranked;
    ranked.reserve(static_cast<size_t>(count));
    for (llama_token token = 0; token < count; ++token) {
        if (!std::isnan(logits[token])) {
            ranked.emplace_back(logits[token], token);
        }
    }
    const size_t top_count = std::min<size_t>(5, ranked.size());
    std::partial_sort(
            ranked.begin(),
            ranked.begin() + static_cast<std::ptrdiff_t>(top_count),
            ranked.end(),
            [](const auto & left, const auto & right) {
                return left.first > right.first ||
                       (left.first == right.first && left.second < right.second);
            });
    static size_t position = 0;
    std::fprintf(stderr, "llama-logits: position=%zu sampled=%d top=", position, sampled);
    for (size_t index = 0; index < top_count; ++index) {
        std::fprintf(
                stderr,
                "%s%d:%.9g",
                index == 0 ? "" : ",",
                ranked[index].second,
                ranked[index].first);
    }
    std::fputc('\n', stderr);
    ++position;
}

static llama_token leone_oracle_sample(
        common_sampler * sampler,
        llama_context * context,
        int index) {
    const llama_token token = common_sampler_sample(sampler, context, index);
    if (std::getenv("LEONE_LLAMA_TOKEN_LOG") != nullptr) {
        std::fprintf(stderr, "leone-token: %d\n", token);
    }
    leone_oracle_log_logits(context, token);
    return token;
}

#define common_sampler_sample(sampler, context, index) \
    leone_oracle_sample((sampler), (context), (index))
#include "../external/llama.cpp/tools/completion/completion.cpp"
