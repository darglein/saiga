/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "post_processing_vertex_shader.glsl"


##GL_FRAGMENT_SHADER
#version 330

#include "camera.glsl"
#include "post_processing_helper_fs.glsl"




uniform int uBlurSize = 4; // use size of noise texture



void main() {
        vec2 texelSize = 1.0 / vec2(textureSize(image, 0));

//	ideally use a fixed size noise and blur so that this loop can be unrolled
        vec4 fResult = vec4(0.0);
        vec2 hlim = vec2(float(-uBlurSize) * 0.5 + 0.5);
        for (int x = 0; x < uBlurSize; ++x) {
                for (int y = 0; y < uBlurSize; ++y) {
                        vec2 offset = vec2(float(x), float(y));
                        offset += hlim;
                        offset *= texelSize;

                        fResult += texture(image, tc + offset);
                }
        }

        fResult = fResult / (uBlurSize * uBlurSize);
        out_color = fResult;
}
