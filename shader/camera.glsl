/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



layout (std140) uniform cameraData
{
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    vec4 camera_position;
};
