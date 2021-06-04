/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/overlay/Layout.h"

namespace Saiga
{
Layout::Layout(int width, int height) : width(width), height(height)
{
    init(width, height);
}

void Layout::init(int _width, int _height)
{
    width  = _width;
    height = _height;

    aspect       = (float)width / (float)height;
    targetWidth  = aspect;
    targetHeight = 1.0f;
    scale        = vec3(aspect, 1.0f, 1.0f);


    //    proj = ortho(0.0f,this->targetWidth,0.0f,this->targetHeight,-1.0f,1.0f);

    cam.setProj(0.0f, targetWidth, 0.0f, targetHeight, -1.0f, 1.0f);
    proj = cam.proj;
}

AABB Layout::transform(Object3D* obj, const AABB& box, vec2 relPos, float relSize, Alignment alignmentX,
                       Alignment alignmentY, bool scaleX)
{
    vec3 s = box.max - box.min;
    // scale to correct size
    if (scaleX)
    {
        float ds = relSize / s[1];
        s        = make_vec3(ds);
        s[0] *= 1.0f / aspect;
    }
    else
    {
        float ds = relSize / s[0];
        s        = make_vec3(ds);
        s[1] *= 1.0f / aspect;
    }
    obj->setScale(s);


    // alignment
    vec3 center = (box.getPosition().array(), obj->getScale().array());
    vec3 bbmin  = (box.min.array() * obj->getScale().array());
    vec3 bbmax  = (box.max.array() * obj->getScale().array());


    vec3 alignmentOffset = make_vec3(0);

    switch (alignmentX)
    {
        case LEFT:
            alignmentOffset[0] += bbmin[0];
            break;
        case RIGHT:
            alignmentOffset[0] += bbmax[0];
            break;
        case CENTER:
            alignmentOffset[0] += center[0];
            break;
    }

    switch (alignmentY)
    {
        case LEFT:
            alignmentOffset[1] += bbmin[1];
            break;
        case RIGHT:
            alignmentOffset[1] += bbmax[1];
            break;
        case CENTER:
            alignmentOffset[1] += center[1];
            break;
    }


    obj->setPosition(vec3(make_vec3(relPos, 0) - alignmentOffset));
    //    std::cout << "obj position " << relPos << " " << alignmentOffset << " " << obj->position << std::endl;

    AABB resultBB = AABB((box.min.array() * s.array()), (box.max.array() * s.array()));
    resultBB.setPosition(obj->getPosition() + center);

    obj->multScale(scale);
    obj->position = (obj->position.array() * make_vec4(scale, 1).array());

    obj->calculateModel();


    return resultBB;
}

AABB Layout::transformNonUniform(Object3D* obj, const AABB& box, vec2 relPos, vec2 relSize,
                                 Layout::Alignment alignmentX, Layout::Alignment alignmentY)
{
    vec3 s = box.max - box.min;
    s      = (vec3(relSize[0], relSize[1], 1.0f).array() / vec3(s[0], s[1], 1.0f).array());
    obj->setScale(s);


    // alignment
    vec3 center = (box.getPosition().array() * obj->getScale().array());
    vec3 bbmin  = (box.min.array() * obj->getScale().array());
    vec3 bbmax  = (box.max.array() * obj->getScale().array());

    vec3 alignmentOffset(0, 0, 0);

    switch (alignmentX)
    {
        case LEFT:
            alignmentOffset[0] += bbmin[0];
            break;
        case RIGHT:
            alignmentOffset[0] += bbmax[0];
            break;
        case CENTER:
            alignmentOffset[0] += center[0];
            break;
    }

    switch (alignmentY)
    {
        case LEFT:
            alignmentOffset[1] += bbmin[1];
            break;
        case RIGHT:
            alignmentOffset[1] += bbmax[1];
            break;
        case CENTER:
            alignmentOffset[1] += center[1];
            break;
    }


    obj->setPosition(vec3(make_vec3(relPos, 0) - alignmentOffset));

    AABB resultBB = AABB((box.min.array() * s.array()), (box.max.array() * s.array()));
    resultBB.setPosition(obj->getPosition() + center);

    obj->multScale(scale);
    obj->position = (obj->position.array() * make_vec4(scale, 1).array());
    obj->calculateModel();

    return resultBB;
}

AABB Layout::transformUniform(Object3D* obj, const AABB& box, vec2 relPos, vec2 relSize, Layout::Alignment alignmentX,
                              Layout::Alignment alignmentY)
{
    relSize[0] *= aspect;
    vec3 s = box.max - box.min;


    //    s[0] *= aspect;
    s = (vec3(relSize[0], relSize[1], 1.0f).array() / vec3(s[0], s[1], 1.0f).array());

    //    std::cout << "s: " << s << std::endl;
    //    std::cout << "test: " << (s * (box.max-box.min)) << " " << relSize << std::endl;

    // use lower value of s[0] and s[1] to scale uniformly.
    //-> The result will fit in the box
    float ds = std::min(s[0], s[1]);

    obj->setScale(vec3(ds, ds, 1));
    obj->scale[0] *= 1.0f / aspect;

    s = obj->getScale();

    // alignment
    vec3 center = (box.getPosition().array(), obj->getScale().array());
    vec3 bbmin  = (box.min.array() * obj->getScale().array());
    vec3 bbmax  = (box.max.array() * obj->getScale().array());

    vec3 alignmentOffset(0, 0, 0);

    switch (alignmentX)
    {
        case LEFT:
            alignmentOffset[0] += bbmin[0];
            break;
        case RIGHT:
            alignmentOffset[0] += bbmax[0];
            break;
        case CENTER:
            alignmentOffset[0] += center[0];
            break;
    }

    switch (alignmentY)
    {
        case LEFT:
            alignmentOffset[1] += bbmin[1];
            break;
        case RIGHT:
            alignmentOffset[1] += bbmax[1];
            break;
        case CENTER:
            alignmentOffset[1] += center[1];
            break;
    }


    obj->setPosition(vec3(make_vec3(relPos, 0) - alignmentOffset));

    AABB resultBB = AABB((box.min.array() * s.array()), (box.max.array() * s.array()));
    resultBB.setPosition(obj->getPosition() + center);

    obj->multScale(scale);
    obj->position = (obj->position.array() * make_vec4(scale, 1).array());
    obj->calculateModel();

    return resultBB;
}

vec2 Layout::transformToLocal(vec2 p)
{
    return vec2(p[0] / width * targetWidth, p[1] / height * targetHeight);
}

vec2 Layout::transformToLocalNormalized(vec2 p)
{
    return vec2(p[0] / width, p[1] / height);
}

}  // namespace Saiga
