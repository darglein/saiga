#include "saiga/util/color.h"

Color::Color(int r, int g, int b, int a)
    : Color((u_int8_t)r, (u_int8_t)g, (u_int8_t)b, (u_int8_t)a)
{

}

Color::Color(u_int8_t r, u_int8_t g, u_int8_t b, u_int8_t a)
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

