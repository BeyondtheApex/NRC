#include "nrc/nrc_reference_nn.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

bool require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}

bool near(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1.0e-5f;
}

void set_hidden_identity(std::vector<float> &weights) {
    for (std::uint32_t layer = 0; layer < nrc::kNnHiddenLayerCount; ++layer) {
        const std::size_t offset = nrc::layer_weight_offset(layer);
        for (std::uint32_t i = 0; i < nrc::kNnHiddenWidth; ++i) {
            weights[offset + i * nrc::kNnHiddenWidth + i] = 1.0f;
        }
    }
}

} // namespace

int main() {
    bool ok = true;

    ok &= require(nrc::layer_weight_offset(0) == 0, "layer 0 offset");
    ok &= require(nrc::layer_weight_offset(1) == 4096, "layer 1 offset");
    ok &= require(nrc::layer_weight_offset(5) == 20480, "output layer offset");

    std::vector<float> weights(nrc::weight_count(), 0.0f);
    nrc::ReferenceInput input{};
    input[0] = 2.0f;
    input[1] = -3.0f;
    input[2] = 4.0f;

    nrc::ReferenceOutput output{};
    nrc::reference_forward(weights, input, output);
    ok &= require(near(output[0], 0.0f) && near(output[1], 0.0f) && near(output[2], 0.0f),
                  "zero weights produce zero output");

    set_hidden_identity(weights);
    const std::size_t outputOffset = nrc::layer_weight_offset(nrc::kNnHiddenLayerCount);
    weights[outputOffset + 0 * nrc::kNnHiddenWidth + 0] = 1.0f;
    weights[outputOffset + 1 * nrc::kNnHiddenWidth + 1] = 1.0f;
    weights[outputOffset + 2 * nrc::kNnHiddenWidth + 2] = 0.5f;

    nrc::reference_forward(weights, input, output);
    ok &= require(near(output[0], 2.0f), "identity hidden layer keeps positive activation");
    ok &= require(near(output[1], 0.0f), "hidden ReLU clamps negative activation");
    ok &= require(near(output[2], 2.0f), "output layer applies row-major weights");

    bool threw = false;
    try {
        nrc::reference_forward(std::span<const float>{weights.data(), weights.size() - 1}, input, output);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    ok &= require(threw, "invalid weight count throws");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
