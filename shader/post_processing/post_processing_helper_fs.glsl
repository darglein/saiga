/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

//uniforms of class PostProcessingShader, see postProcessor.h for more infos
uniform vec4 screenSize;
uniform sampler2D image;
uniform sampler2D gbufferDepth;
uniform sampler2D gbufferNormals;
uniform sampler2D gbufferColor;
uniform sampler2D gbufferData;

in vec2 tc;

layout(location=0) out vec4 out_color;


float linearDepth(float d, float nearPlane, float farPlane){
    float f = farPlane;
    float n = nearPlane;
    return(2 * n) / (f + n - d * (f - n));
}
