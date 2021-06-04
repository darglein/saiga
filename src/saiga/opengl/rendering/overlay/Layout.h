/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/core/geometry/aabb.h"
#include "saiga/core/geometry/object3d.h"

namespace Saiga
{
/**
 * @brief The Layout class
 *
 * Manages the GUI layout, so that different resolutions look good.
 *
 * Concept:
 * You pass in the bounding box of a GUI element and get back a transformation matrix,
 * that moves and scales the object to the right position.
 *
 * The output should be used with an orthogonal camera, with the bounds (0,0,0) (1,1,1)
 */

class SAIGA_OPENGL_API Layout
{
   public:
    enum Alignment
    {
        LEFT   = 0,
        RIGHT  = 1,
        CENTER = 2,
        TOP    = 1,
        BOTTOM = 0
    };

    // private:
    //    static int width,height;
    //    static float targetWidth, targetHeight;
    //    static float aspect;
    // public:

    //    static void init(int width, int height, float targetWidth,  float targetHeight);
    //    static void transform(Object3D* obj, const AABB &box, vec2 relPos, float relSize, Alignment alignmentX,
    //    Alignment alignmentY);

   private:
    int width, height;
    float targetWidth, targetHeight;
    float aspect;
    vec3 scale;

   public:
    mat4 proj;
    OrthographicCamera cam;

    Layout(int width, int height);
    void init(int width, int height);

    // transforms the object uniformly to fit relSize.
    AABB transform(Object3D* obj, const AABB& box, vec2 relPos, float relSize, Alignment alignmentX,
                   Alignment alignmentY, bool scaleX = true);

    // transofrms the obejct to fit relsize
    AABB transformNonUniform(Object3D* obj, const AABB& box, vec2 relPos, vec2 relSize, Alignment alignmentX,
                             Alignment alignmentY);

    // transforms the object uniformly to fit relSize. The resulting object may not cover the complete box defined by
    // relsize.
    AABB transformUniform(Object3D* obj, const AABB& box, vec2 relPos, vec2 relSize, Alignment alignmentX,
                          Alignment alignmentY);

    // transforms a point in the range [(0,0),(width,height)] to the range [(0,0),(targetWidth,targetHeight)]
    vec2 transformToLocal(vec2 p);

    // transforms a point in the range [(0,0),(width,height)] to the range [(0,0),(1,1)]
    vec2 transformToLocalNormalized(vec2 p);
};

}  // namespace Saiga
