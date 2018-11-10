/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/color.h"
#include <algorithm>

#include "internal/noGraphicsAPI.h"
namespace Saiga {

Color::Color() : r(255), g(255),b(255),a(255){

}

Color::Color(int r, int g, int b, int a)
    : r(r), g(g), b(b), a(a)
{

}


Color::Color(float r, float g, float b, float a)
    : Color(vec4(r,g,b,a))
{
}

Color::Color(vec3 c)
    : Color(vec4(c,1))
{

}

Color::Color(vec4 c)
{
    c = round(c * 255.0f);
    r = c.r;
    g = c.g;
    b = c.b;
    a = c.a;
}

Color::operator vec3() const
{
    return toVec3();
}


Color::operator vec4() const
{
    return toVec4();
}

vec3 Color::toVec3() const
{
    return vec3(r/255.0f,g/255.0f,b/255.0f);
}

vec4 Color::toVec4() const
{
    return vec4(r/255.0f,g/255.0f,b/255.0f,a/255.0f);
}

vec3 Color::srgb2linearrgb(vec3 c)
{
    if ( c.x > 0.04045 ) c.x =  pow( ( c.x + 0.055 ) / 1.055 , 2.4);
    else                   c.x = c.x / 12.92;
    if ( c.y > 0.04045 ) c.y = pow( ( c.y + 0.055 ) / 1.055 , 2.4);
    else                   c.y = c.y / 12.92;
    if ( c.z > 0.04045 ) c.z = pow( ( c.z + 0.055 ) / 1.055 , 2.4);
    else                   c.z = c.z / 12.92;
    return c;
}

vec3 Color::linearrgb2srgb(vec3 c)
{
    if ( c.x > 0.0031308 ) c.x = 1.055 *  pow( c.x , ( 1.0f / 2.4f ) ) - 0.055;
    else                     c.x = 12.92 * c.x;
    if ( c.y > 0.0031308 ) c.y = 1.055 * pow( c.y , ( 1.0f / 2.4f ) ) - 0.055;
    else                     c.y = 12.92 * c.y;
    if ( c.z > 0.0031308 ) c.z = 1.055 * pow( c.z , ( 1.0f / 2.4f ) ) - 0.055;
    else                     c.z = 12.92 * c.z;
    return c;
}


vec3 Color::xyz2linearrgb(vec3 c)
{
    vec3 rgbLinear;
    rgbLinear.x = c.x *  3.2406 + c.y * -1.5372 + c.z * -0.4986;
    rgbLinear.y = c.x * -0.9689 + c.y *  1.8758 + c.z *  0.0415;
    rgbLinear.z = c.x *  0.0557 + c.y * -0.2040 + c.z *  1.0570;
    return rgbLinear;
}

vec3 Color::linearrgb2xyz(vec3 c)
{
    //Observer. = 2°, Illuminant = D65
    float X = c.x * 0.4124 + c.y * 0.3576 + c.z * 0.1805;
    float Y = c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
    float Z = c.x * 0.0193 + c.y * 0.1192 + c.z * 0.9505;
    return vec3(X,Y,Z);
}


vec3 Color::rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);

    vec4 p = mix(vec4(c.b,c.g, K.w,K.z), vec4(c.g,c.b, K.x,K.y), mix(1.f, 0.f, c.b < c.g));
    vec4 q = mix(vec4(p.x,p.y,p.w, c.r), vec4(c.r, p.y,p.z,p.x), mix(1.f, 0.f, p.x < c.r));

    float d = q.x - std::min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 Color::hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(vec3(c.x) + vec3(K)) * 6.0f - vec3(K.w));
    return c.z * mix(vec3(K.x), clamp(p - vec3(K.x), vec3(0.0), vec3(1.0)), c.y);
}

}
