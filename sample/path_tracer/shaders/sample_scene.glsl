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

struct NrcSampleSurface {
    vec3 position;
    vec2 scatteredDir;
    vec3 normal;
    float roughness;
    vec3 diffuse;
    vec3 specular;
};

struct NrcSampleRay {
    vec3 origin;
    vec3 direction;
};

struct NrcSampleHit {
    uint primitiveId;
    vec2 barycentric2;
    float t;
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

NrcSampleRay NrcSampleMakePrimaryRay(uvec2 pixel, uvec2 extent) {
    vec2 uv = (vec2(pixel) + vec2(0.5)) / vec2(extent);
    vec2 ndc = uv * 2.0 - vec2(1.0);

    NrcSampleRay ray;
    ray.origin = vec3(0.0, 0.0, -2.0);
    ray.direction = normalize(vec3(ndc, 0.0) - ray.origin);
    return ray;
}

bool NrcSampleIntersectTriangle(NrcSampleRay ray, uint primitiveId, out NrcSampleHit hit) {
    NrcSampleSceneTriangle triangle = nrcSampleSceneTriangles[primitiveId];
    vec3 p0 = triangle.v0.position.xyz;
    vec3 p1 = triangle.v1.position.xyz;
    vec3 p2 = triangle.v2.position.xyz;

    vec3 edge1 = p1 - p0;
    vec3 edge2 = p2 - p0;
    vec3 pvec = cross(ray.direction, edge2);
    float det = dot(edge1, pvec);
    if (abs(det) < 0.000001) {
        return false;
    }

    float invDet = 1.0 / det;
    vec3 tvec = ray.origin - p0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0 || u > 1.0) {
        return false;
    }

    vec3 qvec = cross(tvec, edge1);
    float v = dot(ray.direction, qvec) * invDet;
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }

    float t = dot(edge2, qvec) * invDet;
    if (t <= 0.0) {
        return false;
    }

    hit.primitiveId = primitiveId;
    hit.barycentric2 = vec2(u, v);
    hit.t = t;
    return true;
}

bool NrcSampleTracePrimaryHit(uvec2 pixel, uvec2 extent, out NrcSampleHit hit) {
    NrcSampleRay ray = NrcSampleMakePrimaryRay(pixel, extent);
    bool found = false;
    NrcSampleHit candidate;
    hit.t = 3.402823466e38;

    for (uint primitiveId = 0u; primitiveId < NRC_SAMPLE_SCENE_TRIANGLE_COUNT; ++primitiveId) {
        if (NrcSampleIntersectTriangle(ray, primitiveId, candidate) && candidate.t < hit.t) {
            hit = candidate;
            found = true;
        }
    }

    return found;
}

PackedNrcInput NrcSamplePackHit(uint primitiveId, uint instanceId, vec2 barycentric2, vec2 scatteredDir) {
    PackedNrcInput packedInput;
    packedInput.primitiveId = primitiveId;
    packedInput.flipBitInstanceId = instanceId & 0x7fffffffu;
    packedInput.barycentric2x16U = packUnorm2x16(clamp(barycentric2, vec2(0.0), vec2(1.0)));
    packedInput.scatteredDir2x16U = packUnorm2x16(clamp(scatteredDir, vec2(0.0), vec2(1.0)));
    return packedInput;
}

bool NrcSampleTracePrimaryPackedInput(uvec2 pixel, uvec2 extent, out PackedNrcInput packedInput) {
    vec2 uv = (vec2(pixel) + vec2(0.5)) / vec2(extent);
    NrcSampleHit hit;
    if (!NrcSampleTracePrimaryHit(pixel, extent, hit)) {
        return false;
    }

    packedInput = NrcSamplePackHit(hit.primitiveId,
                                   hit.primitiveId,
                                   hit.barycentric2,
                                   clamp(uv, vec2(0.0), vec2(1.0)));
    return true;
}

PackedNrcInput NrcSampleMakePackedInputFromPixel(uvec2 pixel, uvec2 extent) {
    PackedNrcInput packedInput;
    if (NrcSampleTracePrimaryPackedInput(pixel, extent, packedInput)) {
        return packedInput;
    }
    return NrcSamplePackHit(0u, 0u, vec2(0.0), vec2(0.5));
}

NrcSampleSurface NrcSampleLoadSurface(PackedNrcInput packedInput) {
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

    NrcSampleSurface surface;
    surface.position = position.xyz;
    surface.scatteredDir = scatteredDir;
    surface.normal = normal;
    surface.roughness = clamp(normalRoughness.w, 0.0, 1.0);
    surface.diffuse = clamp(diffuse.rgb, vec3(0.0), vec3(1.0));
    surface.specular = clamp(specular.rgb, vec3(0.0), vec3(1.0));
    return surface;
}

void NrcSampleDirectLightingPayload(PackedNrcInput packedInput, out vec3 bias, out vec3 factor) {
    NrcSampleSurface surface = NrcSampleLoadSurface(packedInput);
    vec3 lightDir = vec3(0.0, 0.0, 1.0);
    vec3 lightColor = vec3(1.4, 1.1, 0.9);
    float nDotL = max(dot(surface.normal, lightDir), 0.0);

    bias = surface.diffuse * lightColor * nDotL;
    factor = max(surface.diffuse + surface.specular * (1.0 - surface.roughness), vec3(0.05));
}

NrcUnpackedInput NrcSampleUnpackPackedInput(PackedNrcInput packedInput) {
    NrcSampleSurface surface = NrcSampleLoadSurface(packedInput);

    NrcUnpackedInput unpackedInput;
    unpackedInput.position = surface.position;
    unpackedInput.scatteredDir = surface.scatteredDir;
    unpackedInput.normal = NrcSphericalEncode(surface.normal);
    unpackedInput.roughness = surface.roughness;
    unpackedInput.diffuse = surface.diffuse;
    unpackedInput.specular = surface.specular;
    return unpackedInput;
}

#endif
