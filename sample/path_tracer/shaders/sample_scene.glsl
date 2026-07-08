#ifndef NRC_SAMPLE_SCENE_GLSL
#define NRC_SAMPLE_SCENE_GLSL

#include "nrc/common/NrcEncoding.glsl"

#ifndef NRC_SAMPLE_SCENE_SET
#define NRC_SAMPLE_SCENE_SET 0
#endif

#ifndef NRC_SAMPLE_SCENE_BINDING
#define NRC_SAMPLE_SCENE_BINDING 3
#endif

#ifndef NRC_SAMPLE_SCENE_TRIANGLE_COUNT
#define NRC_SAMPLE_SCENE_TRIANGLE_COUNT 2u
#endif

struct NrcSampleSceneVertex {
    vec4 position;
    vec4 normalRoughness;
    vec4 diffuse;
    vec4 specular;
};

struct NrcSampleSceneTriangle {
    NrcSampleSceneVertex v0;
    NrcSampleSceneVertex v1;
    NrcSampleSceneVertex v2;
};

layout(std430, set = NRC_SAMPLE_SCENE_SET, binding = NRC_SAMPLE_SCENE_BINDING) readonly buffer NrcSampleSceneBuffer {
    NrcSampleSceneTriangle nrcSampleSceneTriangles[];
};

vec4 NrcSampleBarycentricMix(vec4 v0, vec4 v1, vec4 v2, vec2 barycentric2) {
    float w1 = barycentric2.x;
    float w2 = barycentric2.y;
    float w0 = 1.0 - w1 - w2;
    return v0 * w0 + v1 * w1 + v2 * w2;
}

PackedNrcInput NrcSamplePackHit(uint primitiveId, uint instanceId, vec2 barycentric2, vec2 scatteredDir) {
    PackedNrcInput packedInput;
    packedInput.primitiveId = primitiveId;
    packedInput.flipBitInstanceId = instanceId & 0x7fffffffu;
    packedInput.barycentric2x16U = packUnorm2x16(clamp(barycentric2, vec2(0.0), vec2(1.0)));
    packedInput.scatteredDir2x16U = packUnorm2x16(clamp(scatteredDir, vec2(0.0), vec2(1.0)));
    return packedInput;
}

PackedNrcInput NrcSampleMakePackedInputFromPixel(uvec2 pixel, uvec2 extent) {
    vec2 uv = (vec2(pixel) + vec2(0.5)) / vec2(extent);
    uv = clamp(uv, vec2(0.0), vec2(1.0));

    uint primitiveId = (uv.x + uv.y <= 1.0) ? 0u : 1u;
    vec2 barycentric2 = primitiveId == 0u ? uv : vec2(1.0 - uv.x, 1.0 - uv.y);
    vec2 scatteredDir = vec2(0.25 + 0.5 * uv.x, 0.75 - 0.5 * uv.y);

    return NrcSamplePackHit(primitiveId, primitiveId, barycentric2, scatteredDir);
}

NrcUnpackedInput NrcSampleUnpackPackedInput(PackedNrcInput packedInput) {
    uint primitiveId = min(packedInput.primitiveId, NRC_SAMPLE_SCENE_TRIANGLE_COUNT - 1u);
    vec2 barycentric2 = unpackUnorm2x16(packedInput.barycentric2x16U);
    vec2 scatteredDir = unpackUnorm2x16(packedInput.scatteredDir2x16U);
    NrcSampleSceneTriangle triangle = nrcSampleSceneTriangles[primitiveId];

    vec4 position = NrcSampleBarycentricMix(triangle.v0.position,
                                            triangle.v1.position,
                                            triangle.v2.position,
                                            barycentric2);
    vec4 normalRoughness = NrcSampleBarycentricMix(triangle.v0.normalRoughness,
                                                   triangle.v1.normalRoughness,
                                                   triangle.v2.normalRoughness,
                                                   barycentric2);
    vec4 diffuse = NrcSampleBarycentricMix(triangle.v0.diffuse,
                                           triangle.v1.diffuse,
                                           triangle.v2.diffuse,
                                           barycentric2);
    vec4 specular = NrcSampleBarycentricMix(triangle.v0.specular,
                                            triangle.v1.specular,
                                            triangle.v2.specular,
                                            barycentric2);

    vec3 normalRaw = normalRoughness.xyz;
    float normalLength = length(normalRaw);
    vec3 normal = normalLength > 0.0001 ? normalRaw / normalLength : vec3(0.0, 0.0, 1.0);

    NrcUnpackedInput unpackedInput;
    unpackedInput.position = position.xyz;
    unpackedInput.scatteredDir = scatteredDir;
    unpackedInput.normal = NrcSphericalEncode(normal);
    unpackedInput.roughness = clamp(normalRoughness.w, 0.0, 1.0);
    unpackedInput.diffuse = clamp(diffuse.rgb, vec3(0.0), vec3(1.0));
    unpackedInput.specular = clamp(specular.rgb, vec3(0.0), vec3(1.0));
    return unpackedInput;
}

#endif
