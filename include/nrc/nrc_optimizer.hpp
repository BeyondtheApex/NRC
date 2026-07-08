#pragma once

#include "nrc/nrc_constants.hpp"
#include "nrc/nrc_records.hpp"

#include <cstdint>
#include <span>

namespace nrc {

struct OptimizerConfig {
    float learningRate = kOptimizerLearningRate;
    float epsilon = kOptimizerEpsilon;
    float lossScale = kLossScale;
    float adamBeta1 = kAdamBeta1;
    float adamBeta2 = kAdamBeta2;
    float emaAlpha = kEmaAlpha;
};

void initialize_optimizer_entries(std::span<const float> weights,
                                  std::span<NrcOptimizerEntry> entries);

std::uint32_t prepare_training_batch(std::uint32_t requestedCount,
                                     NrcOptimizerState &state,
                                     OptimizerConfig config = {});

void optimize_weights(std::span<const float> gradients,
                      std::uint32_t trainCount,
                      const NrcOptimizerState &state,
                      std::span<NrcOptimizerEntry> entries,
                      std::span<float> weights,
                      std::span<float> useWeights = {},
                      bool useEmaWeights = false,
                      OptimizerConfig config = {});

} // namespace nrc
