#include "nrc/nrc_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nrc {
namespace {

void require_weight_count(std::size_t count, const char *name) {
    if (count != weight_count()) {
        throw std::invalid_argument(std::string(name) + " must match nrc::weight_count()");
    }
}

} // namespace

void initialize_optimizer_entries(std::span<const float> weights,
                                  std::span<NrcOptimizerEntry> entries) {
    require_weight_count(weights.size(), "weights");
    require_weight_count(entries.size(), "entries");

    for (std::size_t i = 0; i < weights.size(); ++i) {
        entries[i] = {};
        entries[i].weight = weights[i];
        entries[i].emaWeight = weights[i];
    }
}

std::uint32_t prepare_training_batch(std::uint32_t requestedCount,
                                     NrcOptimizerState &state,
                                     OptimizerConfig config) {
    const std::uint32_t count = std::min(requestedCount, kTrainBatchSize);
    if (count != 0u) {
        ++state.step;
        state.beta1T *= config.adamBeta1;
        state.beta2T *= config.adamBeta2;
        state.alphaTMinus1 = state.alphaT;
        state.alphaT *= config.emaAlpha;
    }
    return count;
}

void optimize_weights(std::span<const float> gradients,
                      std::uint32_t trainCount,
                      const NrcOptimizerState &state,
                      std::span<NrcOptimizerEntry> entries,
                      std::span<float> weights,
                      std::span<float> useWeights,
                      bool useEmaWeights,
                      OptimizerConfig config) {
    require_weight_count(gradients.size(), "gradients");
    require_weight_count(entries.size(), "entries");
    require_weight_count(weights.size(), "weights");
    if (!useWeights.empty()) {
        require_weight_count(useWeights.size(), "useWeights");
    }
    if (trainCount == 0u) {
        return;
    }

    const float invTrainScale = 1.0f / (static_cast<float>(trainCount) * config.lossScale);
    const float beta1Denom = 1.0f - state.beta1T;
    const float beta2Denom = 1.0f - state.beta2T;
    const float etaT = 1.0f - state.alphaT;
    const float etaTMinus1 = 1.0f - state.alphaTMinus1;

    for (std::size_t i = 0; i < gradients.size(); ++i) {
        float gradient = gradients[i] * invTrainScale;
        if (!std::isfinite(gradient)) {
            gradient = 0.0f;
        }

        NrcOptimizerEntry entry = entries[i];
        entry.moment1 = config.adamBeta1 * entry.moment1 + (1.0f - config.adamBeta1) * gradient;
        entry.moment2 = config.adamBeta2 * entry.moment2 + (1.0f - config.adamBeta2) * gradient * gradient;

        const float correctedMoment1 = entry.moment1 / beta1Denom;
        const float correctedMoment2 = entry.moment2 / beta2Denom;
        entry.weight -= config.learningRate * correctedMoment1 /
                        (std::sqrt(correctedMoment2) + config.epsilon);

        entry.emaWeight = (1.0f - config.emaAlpha) / etaT * entry.weight +
                          config.emaAlpha * etaTMinus1 * entry.emaWeight;

        entries[i] = entry;
        weights[i] = entry.weight;
        if (!useWeights.empty()) {
            useWeights[i] = useEmaWeights ? entry.emaWeight : entry.weight;
        }
    }
}

} // namespace nrc
