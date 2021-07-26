/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec3 in_position;
layout(location=1) in vec4 in_color;
layout(location=2) in float data;

#include "camera.glsl"
uniform mat4 model;

out vec4 color;
out float radius;

void main() {
    gl_Position = view *model* vec4(in_position,1);
    color = vec4(in_color);
    radius = data;
}






##GL_GEOMETRY_SHADER
#version 330

layout(points) in;
layout(triangle_strip, max_vertices=4) out;

//in vec4[1] color;
//in float[1] radius;

in vec4 color[];
in float radius[];

#include "camera.glsl"
uniform mat4 model;

out vec2 tc;
out vec4 color2;
out vec3 vertexMV;

void main() {

  vec4 pos = gl_in[0].gl_Position;
  vec4 coords = gl_in[0].gl_Position;

  float dx=radius[0];
  float dy=radius[0];

  vec4 ix=vec4(-1,1,-1,1);
  vec4 iy=vec4(-1,-1,1,1);
  vec4 tx=vec4(0,1,0,1);
  vec4 ty=vec4(0,0,1,1);


  for(int i =0; i<4;i++){
      pos.x =ix[i]*dx +coords.x;
      pos.y =iy[i]*dy +coords.y;
      tc.x = tx[i];
      tc.y = ty[i];
      color2 = color[0];
      vertexMV = vec3(pos);
      gl_Position = proj*pos;
      EmitVertex();
  }
}




##GL_FRAGMENT_SHADER

#version 330

uniform sampler2D normal_map;

in vec2 tc;
in vec4 color2;
in vec3 vertexMV;


void main() {

    vec2 reltc = tc*2-vec2(1);
//    if(length(reltc)>=1)
//	discard;

    float lensqr = dot(reltc, reltc);
    if(lensqr > 1.0)
	discard;

}


