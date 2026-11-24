/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The cube map directions of the environment map's compute passes (S36, A4; Vulkan's face order +x, -x, +y, -y, +z,
// -z; s to the right, t down).
#ifndef ENV_COMMON_GLSL
#define ENV_COMMON_GLSL

vec3 cubeDirection(uint face, vec2 st)
{
    float s = st.x, t = st.y;
    vec3 d;
    if (face == 0u)
        d = vec3(1.0, -t, -s);
    else if (face == 1u)
        d = vec3(-1.0, -t, s);
    else if (face == 2u)
        d = vec3(s, 1.0, t);
    else if (face == 3u)
        d = vec3(s, -1.0, -t);
    else if (face == 4u)
        d = vec3(s, -t, 1.0);
    else
        d = vec3(-s, -t, -1.0);
    return normalize(d);
}

// The direction of the texel p of a face of size n
vec3 cubeTexelDirection(uint face, ivec2 p, int n)
{
    return cubeDirection(face, (vec2(p) + 0.5) / float(n) * 2.0 - 1.0);
}

#endif
