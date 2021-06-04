/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/color.h"

#include "internal/noGraphicsAPI.h"

#include <algorithm>
namespace Saiga
{
Color::operator vec3() const
{
    return toVec3();
}


Color::operator vec4() const
{
    return toVec4();
}

Color::Color(float r, float g, float b, float a) : Color(vec4(r, g, b, a)) {}

Color::Color(const vec3& c) : Color(make_vec4(c, 1)) {}

Color::Color(const ucvec4& c) : r(c(0)), g(c(1)), b(c(2)), a(c(3)) {}

Color::Color(const vec4& c) : Color(ucvec4((c * 255.0f).array().round().cast<unsigned char>())) {}

Saiga::Color::operator ucvec4() const
{
    return ucvec4(r, g, b, a);
}

vec3 Color::toVec3() const
{
    return vec3(r / 255.0f, g / 255.0f, b / 255.0f);
}

vec4 Color::toVec4() const
{
    return vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);
}

vec3 Color::srgb2linearrgb(vec3 c)
{
    if (c[0] > 0.04045)
        c[0] = pow((c[0] + 0.055) / 1.055, 2.4);
    else
        c[0] = c[0] / 12.92;
    if (c[1] > 0.04045)
        c[1] = pow((c[1] + 0.055) / 1.055, 2.4);
    else
        c[1] = c[1] / 12.92;
    if (c[2] > 0.04045)
        c[2] = pow((c[2] + 0.055) / 1.055, 2.4);
    else
        c[2] = c[2] / 12.92;
    return c;
}

vec3 Color::linearrgb2srgb(vec3 c)
{
    if (c[0] > 0.0031308)
        c[0] = 1.055 * pow(c[0], (1.0f / 2.4f)) - 0.055;
    else
        c[0] = 12.92 * c[0];
    if (c[1] > 0.0031308)
        c[1] = 1.055 * pow(c[1], (1.0f / 2.4f)) - 0.055;
    else
        c[1] = 12.92 * c[1];
    if (c[2] > 0.0031308)
        c[2] = 1.055 * pow(c[2], (1.0f / 2.4f)) - 0.055;
    else
        c[2] = 12.92 * c[2];
    return c;
}


vec3 Color::xyz2linearrgb(vec3 c)
{
    vec3 rgbLinear;
    rgbLinear[0] = c[0] * 3.2406 + c[1] * -1.5372 + c[2] * -0.4986;
    rgbLinear[1] = c[0] * -0.9689 + c[1] * 1.8758 + c[2] * 0.0415;
    rgbLinear[2] = c[0] * 0.0557 + c[1] * -0.2040 + c[2] * 1.0570;
    return rgbLinear;
}

vec3 Color::linearrgb2xyz(vec3 c)
{
    // Observer. = 2°, Illuminant = D65
    float X = c[0] * 0.4124 + c[1] * 0.3576 + c[2] * 0.1805;
    float Y = c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722;
    float Z = c[0] * 0.0193 + c[1] * 0.1192 + c[2] * 0.9505;
    return vec3(X, Y, Z);
}


vec3 Color::rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);

    vec4 p = mix(vec4(c[2], c[1], K[3], K[2]), vec4(c[1], c[2], K[0], K[1]), mix(1.f, 0.f, c[2] < c[1]));
    vec4 q = mix(vec4(p[0], p[1], p[3], c[0]), vec4(c[0], p[1], p[2], p[0]), mix(1.f, 0.f, p[0] < c[0]));

    float d = q[0] - std::min(q[3], q[1]);
    float e = 1.0e-10;
    return vec3(abs(q[2] + (q[3] - q[1]) / (6.0 * d + e)), d / (q[0] + e), q[0]);
}

vec3 Color::hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    //    vec3 fra = fract(make_vec3(c[0]) + make_vec3(K));
    vec3 fra = (make_vec3(c[0]) + make_vec3(K));
    fra      = fra.array() - fra.array().floor();
    vec3 p   = vec3(fra * 6.0f - make_vec3(K[3])).array().abs();
    return c[2] * mix(make_vec3(K[0]), clamp((p - make_vec3(K[0])).eval(), make_vec3(0.0), make_vec3(1.0)), c[1]);
}

}  // namespace Saiga
