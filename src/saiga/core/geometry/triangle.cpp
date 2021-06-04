/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "triangle.h"

#include "saiga/core/math/random.h"

#include "internal/noGraphicsAPI.h"
namespace Saiga
{
void Triangle::ScaleUniform(float f)
{
    auto cen = center().cast<double>();
    a        = ((a.cast<double>() - cen) * f + cen).cast<float>();
    b        = ((b.cast<double>() - cen) * f + cen).cast<float>();
    c        = ((c.cast<double>() - cen) * f + cen).cast<float>();
}



float Triangle::minimalAngle() const
{
    return acos(cosMinimalAngle());
}

float Triangle::cosMinimalAngle() const
{
    return std::max(std::max(cosAngleAtCorner(0), cosAngleAtCorner(1)), cosAngleAtCorner(2));
}

float Triangle::angleAtCorner(int i) const
{
    return acos(cosAngleAtCorner(i));
}

float Triangle::cosAngleAtCorner(int i) const
{
    vec3 center = a;
    vec3 left   = b;
    vec3 right  = c;


    switch (i)
    {
        case 0:
            center = a;
            left   = b;
            right  = c;
            break;
        case 1:
            center = b;
            left   = c;
            right  = a;
            break;
        case 2:
            center = c;
            left   = a;
            right  = b;
            break;
    }

    return dot(normalize(vec3(left - center)), normalize(vec3(right - center)));
}

bool Triangle::isDegenerate() const
{
    for (int i = 0; i < 3; ++i)
    {
        float a = cosAngleAtCorner(i);
        if (a <= -1 || a >= 1) return true;
    }
    return false;
    //    return !std::isfinite(angleAtCorner(0)) || !std::isfinite(angleAtCorner(1)) ||
    //    !std::isfinite(angleAtCorner(2));
}



vec3 Triangle::RandomPointOnSurface() const
{
    auto r  = Random::MatrixUniform<vec2>(0, 1);
    auto r1 = sqrt(r(0));
    auto r2 = r(1);

    vec3 bary((1 - r1), (r1 * (1 - r2)), r2 * r1);
    return InterpolateBarycentric(bary);

    return (1 - r1) * a + (r1 * (1 - r2)) * b + r2 * r1 * c;
}

vec3 Triangle::RandomBarycentric() const
{
    auto r  = Random::MatrixUniform<vec2>(0, 1);
    auto r1 = sqrt(r(0));
    auto r2 = r(1);

    vec3 bary((1 - r1), (r1 * (1 - r2)), r2 * r1);
    return bary;
}

float mag2(const vec3& x)
{
    return x.dot(x);
}

// find distance x0 is from segment x1-x2
static float point_segment_distance(const vec3& x0, const vec3& x1, const vec3& x2)
{
    vec3 dx(x2 - x1);
    float m2 = mag2(dx);
    // find parameter value of closest point on segment+++
    float s12 = (float)(dot(x2 - x0, dx) / m2);
    if (s12 < 0)
    {
        s12 = 0;
    }
    else if (s12 > 1)
    {
        s12 = 1;
    }
    // and find the distance
    return distance(x0, s12 * x1 + (1 - s12) * x2);
}

vec3 Triangle::BarycentricCoordinates(const vec3& x0) const
{
    auto x1 = a;
    auto x2 = b;
    auto x3 = c;
    // first find barycentric coordinates of closest point on infinite plane
    vec3 x13(x1 - x3), x23(x2 - x3), x03(x0 - x3);
    float m13 = mag2(x13), m23 = mag2(x23), d = dot(x13, x23);
    float invdet = 1.f / std::max(m13 * m23 - d * d, 1e-30f);
    float a = dot(x13, x03), b = dot(x23, x03);
    // the barycentric coordinates themselves
    float w23 = invdet * (m23 * a - d * b);
    float w31 = invdet * (m13 * b - d * a);
    float w12 = 1 - w23 - w31;

    return {w12, w23, w31};
}

vec3 Triangle::InterpolateBarycentric(const vec3& bary) const
{
    return bary(0) * a + bary(1) * b + bary(2) * c;
}

float Triangle::Distance(const vec3& x0) const
{
    auto x1 = a;
    auto x2 = b;
    auto x3 = c;
    // first find barycentric coordinates of closest point on infinite plane
    vec3 x13(x1 - x3), x23(x2 - x3), x03(x0 - x3);
    float m13 = mag2(x13), m23 = mag2(x23), d = dot(x13, x23);
    float invdet = 1.f / std::max(m13 * m23 - d * d, 1e-30f);
    float a = dot(x13, x03), b = dot(x23, x03);
    // the barycentric coordinates themselves
    float w23 = invdet * (m23 * a - d * b);
    float w31 = invdet * (m13 * b - d * a);
    float w12 = 1 - w23 - w31;


    if (w23 >= 0 && w31 >= 0 && w12 >= 0)
    {  // if we're inside the triangle
        return distance(x0, w23 * x1 + w31 * x2 + w12 * x3);
    }
    else
    {                 // we have to clamp to one of the edges
        if (w23 > 0)  // this rules out edge 2-3 for us
            return std::min(point_segment_distance(x0, x1, x2), point_segment_distance(x0, x1, x3));
        else if (w31 > 0)  // this rules out edge 1-3
            return std::min(point_segment_distance(x0, x1, x2), point_segment_distance(x0, x2, x3));
        else  // w12 must be >0, ruling out edge 1-2
            return std::min(point_segment_distance(x0, x1, x3), point_segment_distance(x0, x2, x3));
    }
}

std::ostream& operator<<(std::ostream& os, const Triangle& t)
{
    os << "Triangle: " << t.a.transpose() << "," << t.b.transpose() << "," << t.c.transpose();
    return os;
}

}  // namespace Saiga
