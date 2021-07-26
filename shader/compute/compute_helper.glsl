/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


vec2 Texel2UV(vec2 texel_position, ivec2 image_size)
{
    return vec2(texel_position + vec2(0.5, 0.5)) / vec2(image_size);
}