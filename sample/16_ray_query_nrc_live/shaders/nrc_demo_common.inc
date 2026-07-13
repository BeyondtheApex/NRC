#ifndef NRC_DEMO_COMMON_GLSL
#define NRC_DEMO_COMMON_GLSL

#include "nrc/common/NrcEncoding.glsl"

uint NrcDemoPackUnorm2x16(vec2 value) {
    uvec2 packedValue = uvec2(round(clamp(value, vec2(0.0), vec2(1.0)) * 65535.0));
    return packedValue.x | (packedValue.y << 16u);
}

vec2 NrcDemoUnpackUnorm2x16(uint packedValue) {
    return vec2(float(packedValue & 0xffffu), float((packedValue >> 16u) & 0xffffu)) / 65535.0;
}

float NrcDemoHash01(uint value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return float(value & 0x00ffffffu) / float(0x01000000u);
}

PackedNrcInput NrcDemoMakePackedInput(uint index, uint frame, uint width, uint height) {
    uint x = width == 0u ? 0u : index % width;
    uint y = width == 0u ? 0u : index / width;
    vec2 uv = vec2(float(x) + 0.5, float(y) + 0.5) / max(vec2(width, height), vec2(1.0));

    PackedNrcInput inputValue;
    inputValue.primitiveId = index;
    inputValue.flipBitInstanceId = frame & 1u;
    inputValue.barycentric2x16U = NrcDemoPackUnorm2x16(uv);
    inputValue.scatteredDir2x16U = NrcDemoPackUnorm2x16(vec2(NrcDemoHash01(index ^ frame),
                                                            NrcDemoHash01(index * 1664525u + frame)));
    return inputValue;
}

NrcUnpackedInput NrcUnpackPackedInput(PackedNrcInput packedInput) {
    vec2 uv = NrcDemoUnpackUnorm2x16(packedInput.barycentric2x16U);
    vec2 scattered = NrcDemoUnpackUnorm2x16(packedInput.scatteredDir2x16U);
    float wave = sin(float(packedInput.primitiveId & 255u) * 0.073);

    NrcUnpackedInput inputValue;
    inputValue.position = vec3(uv * 2.0 - 1.0, wave);
    inputValue.scatteredDir = scattered;
    inputValue.normal = normalize(vec2(uv.x - 0.5, uv.y - 0.5) + vec2(0.001));
    inputValue.roughness = 0.15 + 0.8 * scattered.x;
    inputValue.diffuse = vec3(0.25 + 0.65 * uv.x, 0.25 + 0.65 * uv.y, 0.25 + 0.35 * scattered.y);
    inputValue.specular = vec3(0.04 + 0.18 * scattered.x);
    return inputValue;
}

vec3 NrcDemoTargetRadiance(PackedNrcInput packedInput) {
    NrcUnpackedInput inputValue = NrcUnpackPackedInput(packedInput);
    float lobe = max(dot(normalize(vec3(inputValue.normal, 0.65)), normalize(vec3(0.35, 0.55, 0.75))), 0.0);
    vec3 color = inputValue.diffuse * (0.15 + 1.8 * lobe);
    color += inputValue.specular * pow(max(inputValue.scatteredDir.x, 0.0), 8.0) * 4.0;
    return color;
}

#endif
