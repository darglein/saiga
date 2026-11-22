/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The shadow factors of the shadow light (S36): a1 only the cloud shadow (A4: the 2D cloud layer, cloudShadowAt of
// atmosphere_common.glsl), a2 the cascades (binding 8) and the horizon map (binding 9). Requires scene_block.glsl, the
// cloud coverage (binding 3) and atmosphere_common.glsl.
#ifndef SHADOWS_GLSL
#define SHADOWS_GLSL

// The cloud layer's shadow at the receiver relPos (camera-relative): S42's hook (its transmittance map)
float cloudShadow(vec3 relPos)
{
    if (lighting.cloudMotion.w < 0.5 || lighting.shadowLight.w == 0.0) return 1.0;
    return cloudShadowAt(cloudCoverage, lighting.cloudMotion.xy, relPos.xz, lighting.atmosphere.x + relPos.y,
                         normalize(lighting.shadowLight.xyz), lighting.clouds, lighting.cloudMotion.z);
}

// The visibility of the shadow light at the receiver (relPos, normal n): cascades x horizon x clouds
float shadowFactor(vec3 relPos, vec3 n)
{
    return cloudShadow(relPos);
}

#endif
