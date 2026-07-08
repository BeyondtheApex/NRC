#include "nrc/nrc_optimizer.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {

bool require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}

bool near(float lhs, float rhs, float epsilon = 1.0e-6f) {
    return std::fabs(lhs - rhs) <= epsilon;
}

float expected_first_adam_step(float weight, float gradient, std::uint32_t trainCount) {
    gradient /= static_cast<float>(trainCount) * nrc::kLossScale;
    const float moment1 = (1.0f - nrc::kAdamBeta1) * gradient;
    const float moment2 = (1.0f - nrc::kAdamBeta2) * gradient * gradient;
    const float correctedMoment1 = moment1 / (1.0f - nrc::kAdamBeta1);
    const float correctedMoment2 = moment2 / (1.0f - nrc::kAdamBeta2);
    return weight - nrc::kOptimizerLearningRate * correctedMoment1 /
                        (std::sqrt(correctedMoment2) + nrc::kOptimizerEpsilon);
}

} // namespace

int main() {
    bool ok = true;

    std::vector<float> initialWeights(nrc::weight_count(), 0.25f);
    initialWeights[0] = 1.0f;
    initialWeights[1] = 2.0f;
    initialWeights[2] = -3.0f;

    std::vector<nrc::NrcOptimizerEntry> entries(nrc::weight_count());
    nrc::initialize_optimizer_entries(initialWeights, entries);
    ok &= require(near(entries[0].weight, 1.0f), "entry 0 weight initialized");
    ok &= require(near(entries[0].emaWeight, 1.0f), "entry 0 ema initialized");
    ok &= require(near(entries[0].moment1, 0.0f), "entry 0 moment1 initialized");
    ok &= require(near(entries[0].moment2, 0.0f), "entry 0 moment2 initialized");

    nrc::NrcOptimizerState state{};
    state.beta1T = 1.0f;
    state.beta2T = 1.0f;
    state.alphaT = 1.0f;

    nrc::NrcOptimizerState unchangedState = state;
    const std::uint32_t zeroCount = nrc::prepare_training_batch(0u, unchangedState);
    ok &= require(zeroCount == 0u, "zero batch remains zero");
    ok &= require(unchangedState.step == 0u, "zero batch does not advance step");
    ok &= require(near(unchangedState.beta1T, 1.0f), "zero batch keeps beta1");

    const std::uint32_t trainCount = nrc::prepare_training_batch(nrc::kTrainBatchSize + 7u, state);
    ok &= require(trainCount == nrc::kTrainBatchSize, "batch count clamps");
    ok &= require(state.step == 1u, "prepare advances step");
    ok &= require(near(state.beta1T, nrc::kAdamBeta1), "prepare advances beta1");
    ok &= require(near(state.beta2T, nrc::kAdamBeta2), "prepare advances beta2");
    ok &= require(near(state.alphaTMinus1, 1.0f), "prepare stores previous alpha");
    ok &= require(near(state.alphaT, nrc::kEmaAlpha), "prepare advances alpha");

    std::vector<float> gradients(nrc::weight_count(), 0.0f);
    gradients[0] = 4.0f;
    gradients[1] = std::numeric_limits<float>::infinity();
    gradients[2] = -2.0f;

    std::vector<float> weights(nrc::weight_count(), 0.0f);
    std::vector<float> useWeights(nrc::weight_count(), 0.0f);
    nrc::optimize_weights(gradients, trainCount, state, entries, weights, useWeights, true);

    const float expected0 = expected_first_adam_step(1.0f, gradients[0], trainCount);
    const float expected2 = expected_first_adam_step(-3.0f, gradients[2], trainCount);
    ok &= require(near(weights[0], expected0), "positive gradient updates weight");
    ok &= require(near(weights[1], 2.0f), "non-finite gradient is ignored");
    ok &= require(near(weights[2], expected2), "negative gradient updates weight");
    ok &= require(near(entries[1].moment1, 0.0f), "non-finite gradient leaves moment1 zero");
    ok &= require(near(entries[1].moment2, 0.0f), "non-finite gradient leaves moment2 zero");
    ok &= require(near(useWeights[0], weights[0]), "first-step ema use weight equals updated weight");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
