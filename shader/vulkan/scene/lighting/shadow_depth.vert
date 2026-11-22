#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The cascades' instance casters (S36a2, A5; ShadowPipelineTable): the instance placement of instance.vert
// (compacted[gl_InstanceIndex] -> pose, offset slot, mesh) with the cascade's camera-relative light view-projection.
// Casters nearer to the light than the cascade's near plane are pancaked onto depth 0 (orthographic: exact except for
// triangles crossing the plane; no depthClamp feature, D17). The material, colour word and uv for the alpha test of
// shadow_alpha.frag. Does not include lighting.glsl (the cascade map is this draw's attachment).

#include "../instance_common.glsl"

layout(std430, set = 0, binding = 10) readonly buffer Compacted
{
    uvec2 compacted[];
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in uint inNormal;
layout(location = 2) in uint inColor;
layout(location = 3) in vec2 inUv;

layout(push_constant) uniform Push
{
    mat4 viewProj;
    float alpha;
    float pad0, pad1, pad2;
}
pc;

layout(location = 0) flat out uint outMaterial;
layout(location = 1) flat out uint outColor;
layout(location = 2) out vec2 outUv;

void main()
{
    uvec2 e = compacted[gl_InstanceIndex];
    Pose pose;
    uint mesh;
    if ((e.x & dynamicBit) != 0u)
    {
        DynamicInstance d = dynamics[e.x & ~dynamicBit];
        pose              = dynamicPose(d, pc.alpha);
        mesh              = d.mesh;
    }
    else
    {
        StaticInstance s = instances[e.x];
        pose             = staticPose(s);
        mesh             = s.meshFlags & 0xFFFFu;
    }
    OffsetEntry o = offsets[e.y & 0xFFFFFFu];
    vec3 p        = placePoint(pose.position, o, rotateQuat(pose.rotation, inPosition * pose.scale));
    outMaterial   = meshes[mesh].material;
    outColor      = inColor;
    outUv         = inUv;
    gl_Position   = pc.viewProj * vec4(p, 1.0);
    gl_Position.z = max(gl_Position.z, 0.0);
}
