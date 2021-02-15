/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "camera.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/CoordinateSystems.h"

#include "internal/noGraphicsAPI.h"

#include <iostream>

#include "controllable_camera.h"

namespace Saiga
{
#define ANG2RAD 3.14159265358979323846 / 180.0



void Camera::setView(const mat4& v)
{
    view = v;
    recalculateMatrices();
    model = inverse(view);

    this->position = model.col(3);
    this->rot      = quat_cast(model);
}

void Camera::setView(const vec3& eye, const vec3& center, const vec3& up)
{
    setView(lookAt(eye, center, up));
}

void Camera::setProj(const mat4& p)
{
    proj = p;
    recalculateMatrices();
}

void Camera::updateFromModel()
{
    view = inverse(model);
    recalculateMatrices();
}

void Camera::recalculateMatrices()
{
    viewProj = proj * view;
}

void Camera::recalculatePlanes()
{
    recalculatePlanesFromMatrices();
}

float Camera::linearDepth(float d) const
{
    float f = zFar;
    float n = zNear;

    float l = d * (f - n);  // 5.5
    l       = f + n - l;    // 1

    return (2 * n) / l;
}

float Camera::nonlinearDepth(float l) const
{
    float f = zFar;
    float n = zNear;

    float d = (2 * n) / l;
    d       = f + n - d;
    d       = d / (f - n);

    return d;
}

float Camera::toViewDepth(float d) const
{
    vec4 a(0, 0, d * 2 - 1, 1);
    a = inverse(proj) * a;
    a /= a[3];
    return a[2];
}

float Camera::toNormalizedDepth(float d) const
{
    vec4 a(0, 0, d, 1);
    a = proj * a;
    a /= a[3];
    return a[2] * 0.5 + 0.5;
}



//-------------------------------


void Camera::recalculatePlanesFromMatrices()
{
    vec3 pointsClipSpace[] = {
        vec3(-1, 1, -1), vec3(1, 1, -1), vec3(-1, -1, -1), vec3(1, -1, -1),

        vec3(-1, 1, 1),  vec3(1, 1, 1),  vec3(-1, -1, 1),  vec3(1, -1, 1),
    };

    mat4 m = inverse(mat4(proj * view));
    for (int i = 0; i < 8; ++i)
    {
        vec4 p      = m * make_vec4(pointsClipSpace[i], 1);
        p           = p / p[3];
        vertices[i] = make_vec3(p);
    }

    // side planes
    planes[0] = Plane(vertices[0], vertices[2], vertices[1]);  // near
    planes[1] = Plane(vertices[4], vertices[5], vertices[7]);  // far

    planes[2] = Plane(vertices[0], vertices[1], vertices[4]);  // top
    planes[3] = Plane(vertices[2], vertices[6], vertices[3]);  // bottom
    planes[4] = Plane(vertices[0], vertices[4], vertices[2]);  // left
    planes[5] = Plane(vertices[1], vertices[3], vertices[7]);  // right
}


vec3 Camera::projectToViewSpace(vec3 worldPosition) const
{
    return make_vec3(view * make_vec4(worldPosition, 1));
}

vec3 Camera::projectToNDC(vec3 worldPosition) const
{
    vec4 p = (viewProj * make_vec4(worldPosition, 1));
    p /= p[3];
    return make_vec3(p);
}

vec2 Camera::projectToScreenSpace(vec3 worldPosition, int w, int h) const
{
    vec3 p  = projectToNDC(worldPosition);
    vec2 ip = make_vec2(p);
    ip      = ip * 0.5f + make_vec2(0.5f);
    //        ip *= vec2(w, h);
    ip[0] *= w;
    ip[1] *= h;
    return ip;
}

vec3 Camera::inverseprojectToWorldSpace(vec2 ip, float depth, int w, int h) const
{
    //        ip /= vec2(w, h);
    ip[0] /= w;
    ip[1] /= h;
    ip      = (ip - make_vec2(0.5f)) * 2.0f;
    vec3 p  = make_vec3(ip, depth);
    vec4 wp = inverse(viewProj) * make_vec4(p, 1);
    wp /= wp[3];
    return make_vec3(wp);
}

void Camera::imgui()
{
    bool changed = false;
    changed |= ImGui::InputFloat("zNear", &zNear);
    changed |= ImGui::InputFloat("zFar", &zFar);
    changed |= ImGui::Checkbox("vulkanTransform", &vulkanTransform);
    if (changed) recomputeProj();
}



std::ostream& operator<<(std::ostream& os, const Camera& ca)
{
    os << ca.name;
    //    os<<"Nearplane= ("<<ca.nw*2<<" x "<<ca.nh*2<<") Farplane= ("<<ca.fw*2<<" x "<<ca.fh*2<<")";
    return os;
}



//===================================================================================================


void PerspectiveCamera::setProj(float _fovy, float _aspect, float _zNear, float _zFar, bool vulkanTransform)
{
    _fovy                 = radians(_fovy);
    this->fovy            = _fovy;
    this->aspect          = _aspect;
    this->zNear           = _zNear;
    this->zFar            = _zFar;
    this->vulkanTransform = vulkanTransform;

    tang = (float)tan(fovy * 0.5);


    recomputeProj();
}

void PerspectiveCamera::imgui()
{
    Camera::imgui();
    bool changed = false;
    changed |= ImGui::InputFloat("fovy", &fovy);
    changed |= ImGui::InputFloat("aspect", &aspect);
    if (changed) recomputeProj();
}

void PerspectiveCamera::recomputeProj()
{
    proj = perspective(fovy, aspect, zNear, zFar);

    if (vulkanTransform)
    {
        proj = GL2VulkanNormalizedImage() * proj;
    }
}

void PerspectiveCamera::recalculatePlanes()
{
    //    vec3 right = vec3(model[0]);
    //    vec3 up    = vec3(model[1]);
    //    vec3 dir   = -vec3(model[2]);

    vec3 right = make_vec3(model.col(0));
    vec3 up    = make_vec3(model.col(1));
    vec3 dir   = make_vec3(-model.col(2));

    vec3 nearplanepos = getPosition() + dir * zNear;
    vec3 farplanepos  = getPosition() + dir * zFar;

    // near plane
    planes[0] = Plane(nearplanepos, -dir);
    // far plane
    planes[1] = Plane(farplanepos, dir);


    float nh = zNear * tang;
    float nw = nh * aspect;
    float fh = zFar * tang;
    float fw = fh * aspect;

    // calcuate 4 corners of nearplane
    vertices[0] = nearplanepos + nh * up - nw * right;
    vertices[1] = nearplanepos + nh * up + nw * right;
    vertices[2] = nearplanepos - nh * up - nw * right;
    vertices[3] = nearplanepos - nh * up + nw * right;
    // calcuate 4 corners of farplane
    vertices[4] = farplanepos + fh * up - fw * right;
    vertices[5] = farplanepos + fh * up + fw * right;
    vertices[6] = farplanepos - fh * up - fw * right;
    vertices[7] = farplanepos - fh * up + fw * right;

    // side planes
    planes[2] = Plane(getPosition(), vertices[1], vertices[0]);  // top
    planes[3] = Plane(getPosition(), vertices[2], vertices[3]);  // bottom
    planes[4] = Plane(getPosition(), vertices[0], vertices[2]);  // left
    planes[5] = Plane(getPosition(), vertices[3], vertices[1]);  // right


    //    vec3 fbr = farplanepos - fh * up + fw * right;
    //    vec3 fbr = farplanepos - fh * up;
    vec3 fbr       = vertices[4];
    vec3 sphereMid = (nearplanepos + farplanepos) * 0.5f;
    float r        = distance(fbr, sphereMid);

    boundingSphere.r   = r;
    boundingSphere.pos = sphereMid;

    //    std::cout<<"recalculatePlanes"<<endl;
    //    std::cout<<zNear<<" "<<zFar<<endl;
    //    std::cout<<sphereMid<<" "<<fbr<<endl;
    //    std::cout<<r<<endl;
}

std::ostream& operator<<(std::ostream& os, const PerspectiveCamera& ca)
{
    os << "Type: Perspective Camera\n";
    os << "Name='" << ca.name << "' Fovy=" << ca.fovy << " Aspect=" << ca.aspect << " zNear=" << ca.zNear
       << " zFar=" << ca.zFar << "\n";
    os << static_cast<const Camera&>(ca);
    return os;
}

//=========================================================================================================================

void OrthographicCamera::setProj(float _left, float _right, float _bottom, float _top, float _near, float _far)
{
    this->left   = _left;
    this->right  = _right;
    this->bottom = _bottom;
    this->top    = _top;
    this->zNear  = _near;
    this->zFar   = _far;

    recomputeProj();
}

void OrthographicCamera::setProj(AABB bb)
{
    setProj(bb.min[0], bb.max[0], bb.min[1], bb.max[1], bb.min[2], bb.max[2]);
}

void OrthographicCamera::imgui()
{
    Camera::imgui();
}

void OrthographicCamera::recomputeProj()
{
    proj = ortho(left, right, bottom, top, zNear, zFar);
}

void OrthographicCamera::recalculatePlanes()
{
    vec3 rightv = make_vec3((model.col(0)));
    vec3 up     = make_vec3((model.col(1)));
    vec3 dir    = make_vec3(-(model.col(1)));

    vec3 nearplanepos = getPosition() + dir * zNear;
    vec3 farplanepos  = getPosition() + dir * zFar;

    // near plane
    planes[0] = Plane(nearplanepos, -dir);
    // far plane
    planes[1] = Plane(farplanepos, dir);


#if 0
    //calcuate 4 corners of nearplane
    vertices[0] = nearplanepos + nh * up - nw * right;
    vertices[1] = nearplanepos + nh * up + nw * right;
    vertices[2] = nearplanepos - nh * up - nw * right;
    vertices[3] = nearplanepos - nh * up + nw * right;
    //calcuate 4 corners of farplane
    vertices[4] = farplanepos + fh * up - fw * right;
    vertices[5] = farplanepos + fh * up + fw * right;
    vertices[6] = farplanepos - fh * up - fw * right;
    vertices[7] = farplanepos - fh * up + fw * right;
#else
    // calcuate 4 corners of nearplane
    vertices[0] = nearplanepos + top * up + left * rightv;
    vertices[1] = nearplanepos + top * up + right * rightv;
    vertices[2] = nearplanepos + bottom * up + left * rightv;
    vertices[3] = nearplanepos + bottom * up + right * rightv;
    // calcuate 4 corners of farplane
    vertices[4] = farplanepos + top * up + left * rightv;
    vertices[5] = farplanepos + top * up + right * rightv;
    vertices[6] = farplanepos + bottom * up + left * rightv;
    vertices[7] = farplanepos + bottom * up + right * rightv;
#endif

    // side planes
    //    planes[2].set(getPosition(),vertices[1],vertices[0]); //top
    //    planes[3].set(getPosition(),vertices[2],vertices[3]); //bottom
    //    planes[4].set(getPosition(),vertices[0],vertices[2]); //left
    //    planes[5].set(getPosition(),vertices[3],vertices[1]); //right
    planes[2] = Plane(vertices[0], up);       // top
    planes[3] = Plane(vertices[3], -up);      // bottom
    planes[4] = Plane(vertices[0], -rightv);  // left
    planes[5] = Plane(vertices[3], rightv);   // right

    //    vec3 fbr = farplanepos - fh * up + fw * right;
    //    vec3 fbr = farplanepos - fh * up;
    vec3 sphereMid = (nearplanepos + farplanepos) * 0.5f;
    float r        = distance(vertices[0], sphereMid);

    boundingSphere.r   = r;
    boundingSphere.pos = sphereMid;
}



std::ostream& operator<<(std::ostream& os, const Saiga::OrthographicCamera& ca)
{
    os << "Type: Orthographic Camera";
    os << "Name='" << ca.name << "' left=" << ca.left << " right=" << ca.right << " bottom=" << ca.bottom
       << " top=" << ca.top << " zNear=" << ca.zNear << " zFar=" << ca.zFar;
    os << static_cast<const Saiga::Camera&>(ca);
    return os;
}

}  // namespace Saiga
