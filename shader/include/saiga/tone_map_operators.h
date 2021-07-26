/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

vec3 TonemapGamma(vec3 color, float gamma)
{
    return pow(color, vec3(gamma));
}

vec3 TonemapReinhard(vec3 x, float gamma)
{
    x = x / (1.0 + x);
    return TonemapGamma(x, gamma);
}

vec3 TonemapUE3(vec3 x)
{
    // Used in Unreal Engine 3 up to 4.14. (I think, might be wrong).
    // They've since moved to ACES for output on a larger variety of devices.
    // Very simple and intented for use with color-lut afterwards.
    return x / (x + 0.187) * 1.035;
}

vec3 TonemapPhotographic(vec3 x, float gamma)
{
    // Simple photographic tonemapper.
    // Suggested by Emil Persson on Beyond3D forum.
    x = 1.0 - exp2(-x);
    return TonemapGamma(x, gamma);
}

vec3 Tonemap_Filmic_UC2DefaultToGamma(vec3 linearColor)
{
    // Uncharted II fixed tonemapping formula.
    // The linear to sRGB conversion is baked in.

    vec3 x = max(vec3(0), linearColor - vec3(0.004));
    return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
}


vec3 Tonemap_Filmic_UC2(vec3 linearColor, float linearWhite, float A, float B, float C, float D, float E, float F)
{
    // Uncharted II configurable tonemapper.

    // A = shoulder strength
    // B = linear strength
    // C = linear angle
    // D = toe strength
    // E = toe numerator
    // F = toe denominator
    // Note: E / F = toe angle
    // linearWhite = linear white point value

    vec3 x     = linearColor;
    vec3 color = ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;

    x          = vec3(linearWhite);
    vec3 white = ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;

    return color / white;
}

vec3 Tonemap_Filmic_UC2Default(vec3 linearColor)
{
    // Uncharted II fixed tonemapping formula.
    // Gives a warm and gritty image, saturated shadows and bleached highlights.

    return Tonemap_Filmic_UC2(linearColor, 11.2, 0.22, 0.3, 0.1, 0.2, 0.01, 0.30);
}


float
luminance( vec3 rgb )
{
    return 0.299*rgb.r + 0.587*rgb.g + 0.114*rgb.b;
}

vec3 TonemapDrago(vec3 x, float gamma)
{
    float ldmax = 100;
    float lwmax = 10;
    float b = 0.7;

    // World luminance
    float lw = luminance(x.rgb);  // frag.a;

    // Applying operator on World Luminance
    float n  = ldmax * 0.01 * log(10.0) / log(lwmax + 1.0);
    float m  = log(lw + 1.0);
    float p  = log(2.0 + (pow((lw / lwmax), log(b) / log(0.5)) * 8.0));
    float ld = n * m / p;


    //Mapping new luminance with RGB values
    x /= lw;
    x = TonemapGamma(x, gamma);
    x *= ld;
    return x;
}
