#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The PrototypeFoliage class in the cascades (S36a2, A5): the leaf cards' alpha of the foliage texture (binding 1 of
// the shadow set) against the block's alpha cutoff (no alpha to coverage: a hard cutoff); the opaque parts pass.

#include "scene_block.glsl"

layout(location = 0) in vec3 inUvLayer;

void main()
{
    if (inUvLayer.z >= 0.0 && texture(foliage, inUvLayer).r < lighting.foliageParams.x) discard;
}
