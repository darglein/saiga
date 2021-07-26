/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

##GL_COMPUTE_SHADER
#version 430

#ifndef INPUT_TYPE
#error asdf
#define INPUT_TYPE rgba16f
#endif

struct TonemapParameters
{
    vec4 white_point_exposure;
    vec4 vignette_coeffs;
    vec2 vignette_offset;
    uint flags;
    float padding;
};

struct TonemapTempParameters
{
    // (r,g,b,l)
    vec4 average_color_luminace;
};

layout (std140, binding = 3) uniform lightDataBlockPoint
{
    TonemapParameters params;
};

layout (std140, binding = 4) buffer lightDataTmpBlockPoint
{
    TonemapTempParameters tmp_params;
};


layout(binding=0, INPUT_TYPE) uniform image2D inputTex;

layout(local_size_x = 16, local_size_y = 16) in;

// ====================================================================================


// Simple Polynomial Model Based on
// Vignette and Exposure Calibration and Compensation
// https://grail.cs.washington.edu/projects/vignette/vign.iccv05.pdf
//
// r_sqared is the (squared) distance to the optical center.
// if coefficients == 0 -> No vignetting exists
float VignetteModel(float r_squared, vec3 coefficients)
{
    float r2 = r_squared;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    return 1.0f + coefficients[0] * r2 + coefficients[1] * r4 + coefficients[2] *r6;
}

vec2 NormalizedUV(ivec2 texel, ivec2 size)
{
    float max_size = max(size[0], size[1]);
    vec2 center_pixel = vec2(size) / 2.f;
    vec2 centered_uv = (vec2(texel) -center_pixel) / max_size * 2;
    return centered_uv;
}


vec3 Tonemap(vec3 hdr_color, ivec2 texel, vec2 normalized_uv){

    float exposure_value =  params.white_point_exposure.w;
    if((params.flags & 1) != 0)
    {
        // Auto Exposure
        float average_luminace = tmp_params.average_color_luminace.w;
        exposure_value = log2( max(average_luminace /  0.33, 1e-4) );
    }
    float exposure_factor = 1.f / exp2(exposure_value);

    vec3 white_point = params.white_point_exposure.xyz;
    if((params.flags & 2) != 0)
    {
        // Auto White Balance
        vec3 avg = tmp_params.average_color_luminace.rgb;
        float alpha = avg[1] / avg[0];
        float beta = avg[1] / avg[2];
        white_point = vec3(alpha,1,beta);
    }

    float r2 = dot(normalized_uv + params.vignette_offset, normalized_uv+ params.vignette_offset);
    float vignette = VignetteModel(r2, params.vignette_coeffs.xyz);

    vec3 color = vignette * exposure_factor * hdr_color;

    // White balance
    color = color * white_point;
    return color;
}

void main() {
    ivec2 texel_position = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(inputTex);
    if (texel_position.x >= image_size.x || texel_position.y >= image_size.y)
    {
        return;
    }

    vec3 hdr_value = imageLoad(inputTex, texel_position).rgb;
    vec3 ldr_value = Tonemap(hdr_value, texel_position, NormalizedUV(texel_position, image_size));
    imageStore(inputTex, texel_position, vec4(ldr_value, 1));
}
