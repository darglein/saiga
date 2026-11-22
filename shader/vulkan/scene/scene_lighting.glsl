/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The lighting block of RenderScene's draws for the legacy S19a light model (the default pipeline variant until the a1c
// flip of S36): the block and the foliage texture of set 1 (lighting/scene_block.glsl; the legacy shaders read its
// first seven vec4: S18's fixed light by default, the prototype's sun, ambient, fog and wind with
// RenderScene::setPrototypeEnvironment).
#include "lighting/scene_block.glsl"
