#pragma once

#include "saiga/rendering/object3d.h"
#include "saiga/geometry/aabb.h"

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

class SAIGA_GLOBAL Layout{
public:
    enum Alignment{
        LEFT = 0,
        RIGHT = 1,
        CENTER = 2,
        TOP = 1,
        BOTTOM = 0
    };

//private:
//    static int width,height;
//    static float targetWidth, targetHeight;
//    static float aspect;
//public:

//    static void init(int width, int height, float targetWidth,  float targetHeight);
//    static void transform(Object3D* obj, const aabb &box, vec2 relPos, float relSize, Alignment alignmentX, Alignment alignmentY);

private:
    int width,height;
    float targetWidth, targetHeight;
    float aspect;
    vec3 scale;
public:
    mat4 proj;

    Layout(int width, int height, float targetWidth=1,  float targetHeight=1);

    //transforms the object uniformly to fit relSize.
    aabb transform(Object3D* obj, const aabb &box, vec2 relPos, float relSize, Alignment alignmentX, Alignment alignmentY, bool scaleX=true);

    //transofrms the obejct to fit relsize
    aabb transformNonUniform(Object3D* obj, const aabb &box, vec2 relPos, vec2 relSize, Alignment alignmentX, Alignment alignmentY);

    //transforms the object uniformly to fit relSize. The resulting object may not cover the complete box defined by relsize.
    aabb transformUniform(Object3D* obj, const aabb &box, vec2 relPos, vec2 relSize, Alignment alignmentX, Alignment alignmentY);

    //transforms a point in the range [(0,0),(width,height)] to the range [(0,0),(targetWidth,targetHeight)]
    glm::vec2 transformToLocal(glm::vec2 p);

    //transforms a point in the range [(0,0),(width,height)] to the range [(0,0),(1,1)]
    glm::vec2 transformToLocalNormalized(glm::vec2 p);
};

