/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The lighting block of the scene draws alone (S36; Saiga::Vulkan::SceneLighting in src/saiga/vulkan/scene/
// SceneLighting.h): binding 0 of set 1 (SceneLightingSet) and the foliage texture (binding 1). Included by every shader
// that reads the block; lighting.glsl adds the other bindings of set 1 and the shading. The shadow and atlas shaders
// (a2, b) include this file alone (their layout has only bindings 0 and 1).
layout(std140, set = 1, binding = 0) uniform SceneLightingBlock
{
    // x: the foliage's alpha cutoff (trees and grass), yzw: unused
    vec4 foliageParams;
    // xyz: the camera position (float), w: MSAA samples
    vec4 camera;
    // xyz: the tree wind phases (1.1, 0.9, 4.0 rad/s, modulo 2 pi), w: the wind strength
    vec4 wind;
    // x: 1 tint by LOD
    vec4 debug;

    // xyz: towards the sun, w: illuminance (lux, top of the atmosphere)
    vec4 sun;
    // xyz: towards the moon, w: illuminance (lux, phase applied)
    vec4 moon;
    // xyz: towards the shadow light, w: 0 none, 1 sun, 2 moon
    vec4 shadowLight;
    // x: effective exposure, y: pre-exposure, z: EV100, w: 1 linear output
    vec4 exposure;
    // x: 0 studio, 1 physical; y: enclosed; z: debug view; w: overlay flags
    uvec4 mode;
    // The studio environment: xyz towards the sun, w intensity
    vec4 studioSun;
    // x: ambient, y: 1 hemisphere, z: clay albedo of the prototype's surfaces
    vec4 studioShading;
    // The active fog: x density, y falloff, z enabled, w the camera height above the fog base
    vec4 fogParams;
    vec4 fogColorLin;
    // xyz: the ambient floor, w: the ambient scale
    vec4 minAmbient;
    vec4 enclosedAmbient;
    // x altitude, y radius, z cloudiness, w shadow fade
    vec4 clouds;
    // xy: the coverage texture's position of the camera (camera xz + cloud offset, wrapped to the tile; m), z: the mean
    // coverage at the cloudiness, w: 1 the cloud layer and its shadows are on
    vec4 cloudMotion;
    // xyz: the rows of the star rotation; [0].w the stars' intensity (0 in daylight), [1].w the full moon's illuminance
    // (lux; the moon disc)
    vec4 starRotation[3];
    // rgb; albedoGround.w: the rock slope (n.y)
    vec4 albedoGround;
    vec4 albedoRock;
    vec4 albedoShore;
    vec4 albedoWater;
    vec4 albedoGrass;
    vec4 albedoLeaves;
    vec4 albedoBark;
    vec4 phases[2];
    // x: the camera height above the sea level, y: 1 a sea level exists
    vec4 sea;
    // x: the Exposure view's centre (EV100), y: its range, z: raw radiance views (probe), w: their scale
    vec4 debugParams;
    // x: the camera height above the sea level (m; the atmosphere's LUTs), y: 1 the environment cube is valid, z: its
    // last mip, w: the sky-view LUT's horizon elevation (rad; skyViewHorizon of the radius it was computed at)
    vec4 atmosphere;
    // S36a2: per cascade the camera-relative light view-projection (xy NDC over its rectangle, z depth 0 near .. 1 far)
    // and params (x texel size m, y the blend band in NDC, z 1 valid, w the edge margin in NDC: weight 0 beyond 1 - w)
    mat4 cascadeMatrix[4];
    vec4 cascadeParams[4];
    // x: cascade count (0: off), y: depth bias (texels), z: normal offset (texels), w: 1 linear comparison filtering
    vec4 shadowParams;
    // S36a2: the horizon map: previous and front map (xy corner - camera (m), z 1 / texel, w layer, -1 none), ref (x, y
    // the maps' reference height - camera y, z blend previous -> front, w resolution), params (x on, y tan(elevation),
    // z use G inside the cascades, w linear filtering)
    vec4 horizonPrev;
    vec4 horizonFront;
    vec4 horizonRef;
    vec4 horizonParams;
}
lighting;

// The foliage texture of the PrototypeFoliage class and the grass (a 1x1 dummy without the prototype environment)
layout(set = 1, binding = 1) uniform sampler2DArray foliage;
