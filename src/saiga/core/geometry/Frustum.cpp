/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Frustum.h"

#include <iostream>

namespace Saiga
{
Frustum::Frustum(const mat4& model, float fovy, float aspect, float zNear, float zFar, bool negativ_z, bool negative_y)
{
    float tang = (float)tan(fovy * 0.5);

    vec3 position = make_vec3(model.col(3));

    vec3 right = make_vec3(model.col(0));
    vec3 up    = make_vec3(model.col(1));
    vec3 dir   = make_vec3(model.col(2));

    if (negative_y)
    {
        up = -up;
    }


    if (negativ_z)
    {
        dir = -dir;
    }

    vec3 nearplanepos = position + dir * zNear;
    vec3 farplanepos  = position + dir * zFar;

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
    planes[2] = Plane(position, vertices[1], vertices[0]);  // top
    planes[3] = Plane(position, vertices[2], vertices[3]);  // bottom
    planes[4] = Plane(position, vertices[0], vertices[2]);  // left
    planes[5] = Plane(position, vertices[3], vertices[1]);  // right


    //    vec3 fbr = farplanepos - fh * up + fw * right;
    //    vec3 fbr = farplanepos - fh * up;
    vec3 fbr       = vertices[4];
    vec3 sphereMid = (nearplanepos + farplanepos) * 0.5f;
    float r        = distance(fbr, sphereMid);

    boundingSphere.r   = r;
    boundingSphere.pos = sphereMid;
}

void Frustum::computePlanesFromVertices() {}

Frustum::IntersectionResult Frustum::pointInFrustum(const vec3& p) const
{
    for (int i = 0; i < 6; i++)
    {
        if (planes[i].distance(p) < 0) return OUTSIDE;
    }
    return INSIDE;
}

Frustum::IntersectionResult Frustum::sphereInFrustum(const Sphere& s) const
{
    IntersectionResult result = INSIDE;
    float distance;

    for (int i = 0; i < 6; i++)
    {
        distance = planes[i].distance(s.pos);
        if (distance >= s.r)
        {
            //            std::cout<<"outside of plane "<<i<<" "<<planes[i]<<endl;
            return OUTSIDE;
        }
        else if (distance > -s.r)
            result = INTERSECT;
    }
    return result;
}

Frustum::IntersectionResult Frustum::pointInSphereFrustum(const vec3& p) const
{
    if (boundingSphere.contains(p))
    {
        return INSIDE;
    }
    else
    {
        return OUTSIDE;
    }
}

Frustum::IntersectionResult Frustum::sphereInSphereFrustum(const Sphere& s) const
{
    if (boundingSphere.intersect(s))
    {
        return INSIDE;
    }
    else
    {
        return OUTSIDE;
    }
}

std::array<Triangle, 12> Frustum::ToTriangleList() const
{
    std::array<Triangle, 12> result;

    // near
    result[0] = Triangle(vertices[0], vertices[3], vertices[1]);
    result[1] = Triangle(vertices[0], vertices[2], vertices[3]);
    // far
    result[2] = Triangle(vertices[4], vertices[5], vertices[7]);
    result[3] = Triangle(vertices[4], vertices[7], vertices[6]);
    // top
    result[4] = Triangle(vertices[0], vertices[5], vertices[4]);
    result[5] = Triangle(vertices[0], vertices[1], vertices[5]);
    // bottom
    result[6] = Triangle(vertices[2], vertices[6], vertices[7]);
    result[7] = Triangle(vertices[3], vertices[2], vertices[7]);
    // left
    result[8] = Triangle(vertices[0], vertices[4], vertices[6]);
    result[9] = Triangle(vertices[2], vertices[0], vertices[6]);
    // right
    result[10] = Triangle(vertices[1], vertices[7], vertices[5]);
    result[11] = Triangle(vertices[1], vertices[3], vertices[7]);

    return result;
}


int Frustum::sideOfPlane(const Plane& plane) const
{
    int positive = 0, negative = 0;
    for (int i = 0; i < 8; ++i)
    {
        float t = plane.distance(vertices[i]);
        if (t > 0)
            positive++;
        else if (t < 0)
            negative++;
        if (positive && negative) return 0;
    }
    return (positive) ? 1 : -1;
}

vec2 Frustum::projectedIntervall(const vec3& d) const
{
    vec2 ret(1000000, -1000000);
    for (int i = 0; i < 8; ++i)
    {
        float t = dot(d, vertices[i]);
        ret[0]  = std::min(ret[0], t);
        ret[1]  = std::max(ret[1], t);
    }
    return ret;
}

bool Frustum::intersectSAT(const Frustum& other) const
{
    // check planes of this camera
    for (int i = 0; i < 6; ++i)
    {
        if (other.sideOfPlane(planes[i]) > 0)
        {  // other is entirely on positive side
            //            std::cout << "plane fail1 " << i << std::endl;
            return false;
        }
    }

    // check planes of other camera
    for (int i = 0; i < 6; ++i)
    {
        if (this->sideOfPlane(other.planes[i]) > 0)
        {  // this is entirely on positive side
            //            std::cout << "plane fail2 " << i << std::endl;
            return false;
        }
    }


    // test cross product of pairs of edges, one from each polyhedron
    // since the overlap of the projected intervall is checked parallel edges doesn't have to be tested
    // -> 6 edges for each frustum
    for (int i = 0; i < 6; ++i)
    {
        auto e1 = this->getEdge(i);
        for (int j = 0; j < 6; ++j)
        {
            auto e2 = other.getEdge(j);
            vec3 d  = cross(e1.first - e1.second, e2.first - e2.second);

            vec2 i1 = this->projectedIntervall(d);
            vec2 i2 = other.projectedIntervall(d);

            if (i1[0] > i2[1] || i1[1] < i2[0]) return false;
        }
    }

    return true;
}

bool Frustum::intersectSAT(const Sphere& s) const
{
    for (int i = 0; i < 6; ++i)
    {
        if (planes[i].distance(s.pos) >= s.r)
        {
            return false;
        }
    }

    for (int i = 0; i < 8; ++i)
    {
        const vec3& v = vertices[i];
        vec3 d        = (v - s.pos).normalized();
        vec2 i1       = this->projectedIntervall(d);
        vec2 i2       = s.projectedIntervall(d);
        if (i1[0] > i2[1] || i1[1] < i2[0]) return false;
    }

    for (int i = 0; i < 12; ++i)
    {
        auto edge          = this->getEdge(i);
        vec3 A             = edge.first;
        vec3 B             = edge.second;
        vec3 AP            = (s.pos - A);
        vec3 AB            = (B - A);
        vec3 closestOnEdge = A + dot(AP, AB) / dot(AB, AB) * AB;

        vec3 d  = (closestOnEdge - s.pos).normalized();
        vec2 i1 = this->projectedIntervall(d);
        vec2 i2 = s.projectedIntervall(d);
        if (i1[0] > i2[1] || i1[1] < i2[0]) return false;
    }

    return true;
}

std::pair<vec3, vec3> Frustum::getEdge(int i) const
{
    switch (i)
    {
        case 0:
            return std::pair<vec3, vec3>(vertices[0], vertices[4]); // nTL - fTL
        case 1:
            return std::pair<vec3, vec3>(vertices[1], vertices[5]); // nTR - fTR
        case 2:
            return std::pair<vec3, vec3>(vertices[2], vertices[6]); // nBL - fBL
        case 3:
            return std::pair<vec3, vec3>(vertices[3], vertices[7]); // nBR - fBR
        case 4:
            return std::pair<vec3, vec3>(vertices[0], vertices[1]); // nTL - nTR
        case 5:
            return std::pair<vec3, vec3>(vertices[0], vertices[2]); // nTL - nBL
        case 6:
            return std::pair<vec3, vec3>(vertices[3], vertices[2]); // nBR - nBL
        case 7:
            return std::pair<vec3, vec3>(vertices[3], vertices[1]); // nBR - nTR
        case 8:
            return std::pair<vec3, vec3>(vertices[4], vertices[5]); // fTL - fTR
        case 9:
            return std::pair<vec3, vec3>(vertices[4], vertices[6]); // fTL - fBL
        case 10:
            return std::pair<vec3, vec3>(vertices[7], vertices[6]); // fBR - fBL
        case 11:
            return std::pair<vec3, vec3>(vertices[7], vertices[5]); // fBR - fTR
        default:
            std::cerr << "Camera::getEdge" << std::endl;
            return std::pair<vec3, vec3>();
    }
}


std::ostream& operator<<(std::ostream& os, const Frustum& frustum)
{
    os << "[Frustum]" << std::endl;
    for (auto p : frustum.planes)
    {
        os << p << std::endl;
    }
    for (auto v : frustum.vertices)
    {
        os << v.transpose() << std::endl;
    }
    return os;
}


}  // namespace Saiga
