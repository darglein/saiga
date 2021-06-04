/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"

namespace Saiga
{
class SAIGA_CORE_API Color
{
   public:
    uint8_t r, g, b, a;

    constexpr Color() : r(255), g(255), b(255), a(255) {}
    constexpr Color(int r, int g, int b, int a = 255) : r(r), g(g), b(b), a(a) {}

    Color(float r, float g, float b, float a = 255);

    Color(const vec3& c);

    Color(const ucvec4& c);
    Color(const vec4& c);

    operator vec3() const;
    operator vec4() const;


    operator ucvec4() const;

    vec3 toVec3() const;
    vec4 toVec4() const;

    /**
     * Almost every image is stored in srgb format nowadays.
     * All monitors asume that the input is in srgb format.
     *
     * So if you just want to display a texture on the screen with opengl NO conversions have to be done.
     *
     * The problem with srgb is, that the intensity values are not stored linearly in memory. (The brightness is also
     * called gamma) For example the gray value (0.2,0.2,0.2) is not twice as bright as (0.1,0.1,0.1). The gamma of srgb
     * is approximately 2.2:  Brightness = value ^ 2.2 With such an exponential curve, the brightness near 0 has higher
     * resolution than the brightness near 1. This fits the perception of colors of the human eye, because shades with
     * low intensities are easier to differentiate.
     *
     * Ok, srgb is not linear, where is the problem?
     *
     * The problem is, that arithmetic operations on srgb colors do not produce the expected result.
     * For example, there are 2 light sources, that want to light one pixel. The correct way is it to add the
     * intensities of both lights up.
     * C = C1 + C2
     * But, because srgb is not linear the intensity does not add up correctly
     * I = C1^2.2 + C2^2.2
     *
     * As a result doing lighting calculations in srgb color space is wrong!
     *
     *
     * For this we want to convert the colors to a linear rgb space before doing any calculations
     * and convert it back when we are finished:
     *
     * cl = srgb2linearrgb(c)
     * cl = doLighting(cl)
     * c = linearrgb2srgb(cl)
     *
     * Fortunately OpenGL offers an automatic way for doing this conversions, by using GL_SRGB8 or GL_SRGB8_ALPHA8 as
     * texture format. From opengl.org:
     * [...] What this means is that the values placed in images of this format are assumed to be stored in the sRGB
     * colorspace. When fetching from sRGB images in Shaders, either through Samplers or images, the values retrieved
     * are converted from the sRGB colors into linear colorspace. Thus, the shader only sees linear values. [...]
     *
     * If you want to write to srgb textures use glEnable(GL_FRAMEBUFFER_SRGB); this will do the conversion from linear
     * to srgb.
     *
     *
     * So all textures are now processed correctly. The last thing to do is to make sure all colors, that are passed as
     * uniforms or vertex-data to the shader are in linear rgb space. For this use the functions below.
     */

    static vec3 srgb2linearrgb(vec3 c);
    static vec3 linearrgb2srgb(vec3 c);

    static vec3 xyz2linearrgb(vec3 c);
    static vec3 linearrgb2xyz(vec3 c);

    static vec3 rgb2hsv(vec3 c);
    static vec3 hsv2rgb(vec3 c);
};

namespace Colors
{
// RGB Color table copied from http://www.rapidtables.com/web/color/RGB_Color.htm
constexpr Color maroon               = Color(128, 0, 0);
constexpr Color darkred              = Color(139, 0, 0);
constexpr Color brown                = Color(165, 42, 42);
constexpr Color firebrick            = Color(178, 34, 34);
constexpr Color crimson              = Color(220, 20, 60);
constexpr Color red                  = Color(255, 0, 0);
constexpr Color tomato               = Color(255, 99, 71);
constexpr Color coral                = Color(255, 127, 80);
constexpr Color indianred            = Color(205, 92, 92);
constexpr Color lightcoral           = Color(240, 128, 128);
constexpr Color darksalmon           = Color(233, 150, 122);
constexpr Color salmon               = Color(250, 128, 114);
constexpr Color lightsalmon          = Color(255, 160, 122);
constexpr Color orangered            = Color(255, 69, 0);
constexpr Color darkorange           = Color(255, 140, 0);
constexpr Color orange               = Color(255, 165, 0);
constexpr Color gold                 = Color(255, 215, 0);
constexpr Color darkgoldenrod        = Color(184, 134, 11);
constexpr Color goldenrod            = Color(218, 165, 32);
constexpr Color palegoldenrod        = Color(238, 232, 170);
constexpr Color darkkhaki            = Color(189, 183, 107);
constexpr Color khaki                = Color(240, 230, 140);
constexpr Color olive                = Color(128, 128, 0);
constexpr Color yellow               = Color(255, 255, 0);
constexpr Color yellowgreen          = Color(154, 205, 50);
constexpr Color darkolivegreen       = Color(85, 107, 47);
constexpr Color olivedrab            = Color(107, 142, 35);
constexpr Color lawngreen            = Color(124, 252, 0);
constexpr Color chartreuse           = Color(127, 255, 0);
constexpr Color greenyellow          = Color(173, 255, 47);
constexpr Color dargreen             = Color(0, 100, 0);
constexpr Color green                = Color(0, 128, 0);
constexpr Color forestgreen          = Color(34, 139, 34);
constexpr Color lime                 = Color(0, 255, 0);
constexpr Color limegreen            = Color(50, 205, 50);
constexpr Color lightgreen           = Color(144, 238, 144);
constexpr Color palegreen            = Color(152, 251, 152);
constexpr Color darkseagreen         = Color(143, 188, 143);
constexpr Color mediumspringgreen    = Color(0, 250, 154);
constexpr Color springgreen          = Color(0, 255, 127);
constexpr Color seagreen             = Color(46, 139, 87);
constexpr Color mediumaquamarine     = Color(102, 205, 170);
constexpr Color mediumseagreen       = Color(60, 179, 113);
constexpr Color lightseagreen        = Color(32, 178, 170);
constexpr Color darkslategray        = Color(47, 79, 79);
constexpr Color teal                 = Color(0, 128, 128);
constexpr Color darkcyan             = Color(0, 139, 139);
constexpr Color aqua                 = Color(0, 255, 255);
constexpr Color cyan                 = Color(0, 255, 255);
constexpr Color lightcyan            = Color(224, 255, 255);
constexpr Color darkturquoise        = Color(0, 206, 209);
constexpr Color turquoise            = Color(64, 224, 208);
constexpr Color mediumturquoise      = Color(72, 209, 204);
constexpr Color paleturquoise        = Color(175, 238, 238);
constexpr Color aquamarine           = Color(127, 255, 212);
constexpr Color powderblue           = Color(176, 224, 230);
constexpr Color cadetblue            = Color(95, 158, 160);
constexpr Color steelblue            = Color(70, 130, 180);
constexpr Color cornflowerblue       = Color(100, 149, 237);
constexpr Color deepskyblue          = Color(0, 191, 255);
constexpr Color dodgerblue           = Color(30, 144, 255);
constexpr Color lightblue            = Color(173, 216, 230);
constexpr Color skyblue              = Color(135, 206, 235);
constexpr Color lightskyblue         = Color(135, 206, 250);
constexpr Color midnightblue         = Color(25, 25, 112);
constexpr Color navy                 = Color(0, 0, 128);
constexpr Color darkblue             = Color(0, 0, 139);
constexpr Color mediumblue           = Color(0, 0, 205);
constexpr Color blue                 = Color(0, 0, 255);
constexpr Color royalblue            = Color(65, 105, 225);
constexpr Color blueviolet           = Color(138, 43, 226);
constexpr Color indigo               = Color(75, 0, 130);
constexpr Color darkslateblue        = Color(72, 61, 139);
constexpr Color slateblue            = Color(106, 90, 205);
constexpr Color mediumslateblue      = Color(123, 104, 238);
constexpr Color mediumpurple         = Color(147, 112, 219);
constexpr Color darkmagenta          = Color(139, 0, 139);
constexpr Color darkviolet           = Color(148, 0, 211);
constexpr Color darkorchid           = Color(153, 50, 204);
constexpr Color mediumorchid         = Color(186, 85, 211);
constexpr Color purple               = Color(128, 0, 128);
constexpr Color thistle              = Color(216, 191, 216);
constexpr Color plum                 = Color(221, 160, 221);
constexpr Color violet               = Color(238, 130, 238);
constexpr Color magenta              = Color(255, 0, 255);
constexpr Color orchid               = Color(218, 112, 214);
constexpr Color mediumvioletred      = Color(199, 21, 133);
constexpr Color palevioletred        = Color(219, 112, 147);
constexpr Color deeppink             = Color(255, 20, 147);
constexpr Color hotpink              = Color(255, 105, 180);
constexpr Color lightpink            = Color(255, 182, 193);
constexpr Color pink                 = Color(255, 192, 203);
constexpr Color antiquewhite         = Color(250, 235, 215);
constexpr Color beige                = Color(245, 245, 220);
constexpr Color bisque               = Color(255, 228, 196);
constexpr Color blanchedalmond       = Color(255, 235, 205);
constexpr Color wheat                = Color(245, 222, 179);
constexpr Color cornsilk             = Color(255, 248, 220);
constexpr Color lemonchiffon         = Color(255, 250, 205);
constexpr Color lightgoldenrodyellow = Color(250, 250, 210);
constexpr Color lightyellow          = Color(255, 255, 224);
constexpr Color saddlebrown          = Color(139, 69, 19);
constexpr Color sienna               = Color(160, 82, 45);
constexpr Color chocolate            = Color(210, 105, 30);
constexpr Color peru                 = Color(205, 133, 63);
constexpr Color sandybrown           = Color(244, 164, 96);
constexpr Color burlywood            = Color(222, 184, 135);
constexpr Color tan                  = Color(210, 180, 140);
constexpr Color rosybrown            = Color(188, 143, 143);
constexpr Color moccasin             = Color(255, 228, 181);
constexpr Color navajowhite          = Color(255, 222, 173);
constexpr Color peachpuff            = Color(255, 218, 185);
constexpr Color mistyrose            = Color(255, 228, 225);
constexpr Color lavenderblush        = Color(255, 240, 245);
constexpr Color linen                = Color(250, 240, 230);
constexpr Color oldlace              = Color(253, 245, 230);
constexpr Color papayawhip           = Color(255, 239, 213);
constexpr Color seashell             = Color(255, 245, 238);
constexpr Color mintcream            = Color(245, 255, 250);
constexpr Color slategray            = Color(112, 128, 144);
constexpr Color lightslategray       = Color(119, 136, 153);
constexpr Color lightsteelblue       = Color(176, 196, 222);
constexpr Color lavender             = Color(230, 230, 250);
constexpr Color floralwhite          = Color(255, 250, 240);
constexpr Color aliceblue            = Color(240, 248, 255);
constexpr Color ghostwhite           = Color(248, 248, 255);
constexpr Color honeydew             = Color(240, 255, 240);
constexpr Color ivory                = Color(255, 255, 240);
constexpr Color azure                = Color(240, 255, 255);
constexpr Color snow                 = Color(255, 250, 250);
constexpr Color black                = Color(0, 0, 0);
constexpr Color dimgray              = Color(105, 105, 105);
constexpr Color gray                 = Color(128, 128, 128);
constexpr Color darkgray             = Color(169, 169, 169);
constexpr Color silver               = Color(192, 192, 192);
constexpr Color lightgray            = Color(211, 211, 211);
constexpr Color gainsboro            = Color(220, 220, 220);
constexpr Color whitesmoke           = Color(245, 245, 245);
constexpr Color white                = Color(255, 255, 255);
}  // namespace Colors

}  // namespace Saiga
