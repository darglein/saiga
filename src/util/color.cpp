#include "saiga/util/color.h"

Color::Color(int r, int g, int b, int a)
    : Color((uint8_t)r, (uint8_t)g, (uint8_t)b, (uint8_t)a)
{

}

Color::Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
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
    c = glm::round(c * 255.0f);
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

vec3 Color::rgb2srgb(vec3 c)
{
    if ( c.x > 0.04045 ) c.x =  glm::pow( ( c.x + 0.055 ) / 1.055 , 2.4);
    else                   c.x = c.x / 12.92;
    if ( c.y > 0.04045 ) c.y = glm::pow( ( c.y + 0.055 ) / 1.055 , 2.4);
    else                   c.y = c.y / 12.92;
    if ( c.z > 0.04045 ) c.z = glm::pow( ( c.z + 0.055 ) / 1.055 , 2.4);
    else                   c.z = c.z / 12.92;
    return c;
}

vec3 Color::rgb2xyz(vec3 c)
{
    if ( c.x > 0.04045 ) c.x =  glm::pow( ( c.x + 0.055 ) / 1.055 , 2.4);
    else                   c.x = c.x / 12.92;
    if ( c.y > 0.04045 ) c.y = glm::pow( ( c.y + 0.055 ) / 1.055 , 2.4);
    else                   c.y = c.y / 12.92;
    if ( c.z > 0.04045 ) c.z = glm::pow( ( c.z + 0.055 ) / 1.055 , 2.4);
    else                   c.z = c.z / 12.92;

//    c.x = c.x * 100;
//    c.y = c.y * 100;
//    c.z = c.z * 100;

    //Observer. = 2°, Illuminant = D65
    float X = c.x * 0.4124 + c.y * 0.3576 + c.z * 0.1805;
    float Y = c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
    float Z = c.x * 0.0193 + c.y * 0.1192 + c.z * 0.9505;
    return vec3(X,Y,Z);
}

vec3 Color::xyz2srgb(vec3 c)
{

    vec3 rgbLinear;
    rgbLinear.x = c.x *  3.2406 + c.y * -1.5372 + c.z * -0.4986;
    rgbLinear.y = c.x * -0.9689 + c.y *  1.8758 + c.z *  0.0415;
    rgbLinear.z = c.x *  0.0557 + c.y * -0.2040 + c.z *  1.0570;

    if ( rgbLinear.x > 0.0031308 ) rgbLinear.x = 1.055 *  glm::pow( rgbLinear.x , ( 1.0f / 2.4f ) ) - 0.055;
    else                     rgbLinear.x = 12.92 * rgbLinear.x;
    if ( rgbLinear.y > 0.0031308 ) rgbLinear.y = 1.055 * glm::pow( rgbLinear.y , ( 1.0f / 2.4f ) ) - 0.055;
    else                     rgbLinear.y = 12.92 * rgbLinear.y;
    if ( rgbLinear.z > 0.0031308 ) rgbLinear.z = 1.055 * glm::pow( rgbLinear.z , ( 1.0f / 2.4f ) ) - 0.055;
    else                     rgbLinear.z = 12.92 * rgbLinear.z;

    return rgbLinear;
}

vec3 Color::srgb2xyz(vec3 c)
{
    return c;
}

