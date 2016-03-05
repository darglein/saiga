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
        RIGHT,
        CENTER
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
public:

    Layout(int width, int height, float targetWidth=1,  float targetHeight=1);
    void transform(Object3D* obj, const aabb &box, vec2 relPos, float relSize, Alignment alignmentX, Alignment alignmentY, bool scaleX=true);
    void transformNonUniform(Object3D* obj, const aabb &box, vec2 relPos, vec2 relSize, Alignment alignmentX, Alignment alignmentY);
};

