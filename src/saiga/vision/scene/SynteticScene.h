/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/scene/Scene.h"

namespace Saiga
{
namespace SynteticScene
{
/**
 * Generates #numWorldPoints 3D points at the origin in a sphere with radius 1.
 * Generates #numCameras cameras equally distributed on a circle in the x-z plane
 * Each camera has #numImagePoints (random) references to world points.
 */
SAIGA_VISION_API Scene CircleSphere(int numWorldPoints, int numCameras, int numImagePoints, bool random_sphere = false);



/**
 * This module generates syntetic Scenes for testing purposes.
 * Checkout the vision/vision_ba sample to see the generated scenes
 */
class SAIGA_VISION_API SceneCreator
{
   public:
    Scene circleSphere();  // use the class members as parameters


    void imgui();

    // Reasonable values after rgbd slam:
    // 40k image point references -> 300-400 points per image
    // 125 key frames
    // 9000 world points
    int numWorldPoints = 9000;
    int numCameras     = 125;
    int numImagePoints = 350;
    bool random_sphere = false;
};
}  // namespace SynteticScene
}  // namespace Saiga
