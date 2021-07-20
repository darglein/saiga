/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;

out vec3 vPosition;

void main()
{
    vPosition = in_position;
}



##GL_TESS_CONTROL_SHADER
#version 330
layout(vertices = 3) out;
in vec3 vPosition[];
out vec3 tcPosition[];
uniform float TessLevelInner;
uniform float TessLevelOuter;


void main()
{
    tcPosition[gl_InvocationID] = vPosition[gl_InvocationID];
    if (gl_InvocationID == 0) {
        gl_TessLevelInner[0] = TessLevelInner;
        gl_TessLevelOuter[0] = TessLevelOuter;
        gl_TessLevelOuter[1] = TessLevelOuter;
        gl_TessLevelOuter[2] = TessLevelOuter;
    }
}


##GL_TESS_EVALUATION_SHADER
#version 330
layout(triangles, equal_spacing, ccw) in;
in vec3 tcPosition[];
out vec3 tePosition;
out vec4 tePositionMV;
out vec3 tePatchDistance;

#include "camera.glsl"
uniform mat4 model;

void main()
{
    vec3 p0 = gl_TessCoord.x * tcPosition[0];
    vec3 p1 = gl_TessCoord.y * tcPosition[1];
    vec3 p2 = gl_TessCoord.z * tcPosition[2];
    tePatchDistance = gl_TessCoord;
    tePosition = normalize(p0 + p1 + p2);
    tePositionMV = view * model * vec4(tePosition, 1);
    gl_Position = proj * tePositionMV;
}


##GL_GEOMETRY_SHADER
#version 330

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
in vec3 tePosition[3];
in vec4 tePositionMV[3];
in vec3 tePatchDistance[3];

out vec4 gVertexMV;
out vec3 gFacetNormal;
out vec3 gPatchDistance;
out vec3 gTriDistance;

void main()
{
    vec3 A = vec3(tePositionMV[2] - tePositionMV[0]);
    vec3 B = vec3(tePositionMV[1] - tePositionMV[0]);
    gFacetNormal = normalize(-cross(A, B));

    gPatchDistance = tePatchDistance[0];
    gVertexMV = tePositionMV[0];
    gTriDistance = vec3(1, 0, 0);
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();

    gPatchDistance = tePatchDistance[1];
    gVertexMV = tePositionMV[1];
    gTriDistance = vec3(0, 1, 0);
    gl_Position = gl_in[1].gl_Position;
    EmitVertex();

    gPatchDistance = tePatchDistance[2];
    gVertexMV = tePositionMV[2];
    gTriDistance = vec3(0, 0, 1);
    gl_Position = gl_in[2].gl_Position;
    EmitVertex();

    EndPrimitive();
}


##GL_FRAGMENT_SHADER

#version 330

in vec4 gVertexMV;
in vec3 gFacetNormal;
in vec3 gPatchDistance;
in vec3 gTriDistance;

#include "geometry_helper_fs.glsl"

float amplify(float d, float scale, float offset)
{
    d = scale * d + offset;
    d = clamp(d, 0, 1);
    d = 1 - exp2(-2*d*d);
    return d;
}

void main() {
    float d1 = min(min(gTriDistance.x, gTriDistance.y), gTriDistance.z);
    float d2 = min(min(gPatchDistance.x, gPatchDistance.y), gPatchDistance.z);

    vec3 color = vec3(1);
    color = amplify(d1, 40, -0.5) * amplify(d2, 60, -0.5) * color;

    setGbufferData(color,gFacetNormal,vec4(0));
}


