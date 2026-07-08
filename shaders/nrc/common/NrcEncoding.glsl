#ifndef NRC_ENCODING_GLSL
#define NRC_ENCODING_GLSL

#include "NrcTypes.glsl"

struct NrcUnpackedInput {
    vec3 position;
    vec2 scatteredDir;
    vec2 normal;
    float roughness;
    vec3 diffuse;
    vec3 specular;
};

vec2 NrcSphericalEncode(vec3 direction) {
    return vec2(direction.xy == vec2(0.0) ? 0.5 : 0.5 + atan(direction.y, direction.x) / (2.0 * NRC_PI),
                acos(clamp(direction.z, -1.0, 1.0)) / NRC_PI);
}

vec4 NrcQuarticCdf(vec4 x, float invRadius) {
    vec4 u = x * invRadius;
    vec4 u2 = u * u;
    vec4 u4 = u2 * u2;
    return clamp((15.0 / 16.0) * u * (1.0 - (2.0 / 3.0) * u2 + (1.0 / 5.0) * u4) + 0.5,
                 vec4(0.0),
                 vec4(1.0));
}

vec4 NrcOneBlob4Encode(float x) {
    vec4 left = vec4(0.0, 0.25, 0.5, 0.75);
    vec4 right = vec4(0.25, 0.5, 0.75, 1.0);
    return NrcQuarticCdf(right - x, 4.0) - NrcQuarticCdf(left - x, 4.0);
}

vec4 NrcTriWave(vec4 x) {
    return 2.0 * abs(mod(x - 0.5, 2.0) - 1.0) - 1.0;
}

mat3x4 NrcFrequencyEncode(float p) {
    mat3x4 f = mat3x4(vec4(1.0, 2.0, 4.0, 8.0),
                      vec4(16.0, 32.0, 64.0, 128.0),
                      vec4(256.0, 512.0, 1024.0, 2048.0)) * p;
    return mat3x4(NrcTriWave(f[0]), NrcTriWave(f[1]), NrcTriWave(f[2]));
}

uvec4 NrcPackHalf8x16(vec4 a, vec4 b) {
    return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}

void NrcEncodeInput(NrcUnpackedInput nrcInput, out uvec4 encoded[NRC_ENCODED_INPUT_UVEC4_COUNT]) {
    mat3x4 posFreq0 = NrcFrequencyEncode(nrcInput.position.x);
    mat3x4 posFreq1 = NrcFrequencyEncode(nrcInput.position.y);
    mat3x4 posFreq2 = NrcFrequencyEncode(nrcInput.position.z);
    vec4 scattered0 = NrcOneBlob4Encode(nrcInput.scatteredDir.x);
    vec4 scattered1 = NrcOneBlob4Encode(nrcInput.scatteredDir.y);
    vec4 normal0 = NrcOneBlob4Encode(nrcInput.normal.x);
    vec4 normal1 = NrcOneBlob4Encode(nrcInput.normal.y);
    vec4 roughness = NrcOneBlob4Encode(1.0 - exp(-nrcInput.roughness));

    encoded[0] = NrcPackHalf8x16(posFreq0[0], posFreq0[1]);
    encoded[1] = NrcPackHalf8x16(posFreq0[2], posFreq1[0]);
    encoded[2] = NrcPackHalf8x16(posFreq1[1], posFreq1[2]);
    encoded[3] = NrcPackHalf8x16(posFreq2[0], posFreq2[1]);
    encoded[4] = NrcPackHalf8x16(posFreq2[2], scattered0);
    encoded[5] = NrcPackHalf8x16(scattered1, normal0);
    encoded[6] = NrcPackHalf8x16(normal1, roughness);
    encoded[7] = NrcPackHalf8x16(vec4(nrcInput.diffuse, nrcInput.specular.r),
                                 vec4(nrcInput.specular.gb, 1.0, 1.0));
}

#endif
