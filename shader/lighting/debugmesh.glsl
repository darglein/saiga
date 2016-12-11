
##GL_VERTEX_SHADER

#version 330
#extension GL_ARB_explicit_attrib_location : enable
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec2 in_tex;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

uniform mat4 MV;
uniform mat4 MVP;

uniform vec4 position;


out vec3 vertexMV;
out vec3 vertex;
out vec3 lightPos;

void main() {
    gl_Position = proj*view *model* vec4(in_position,1);
}





##GL_FRAGMENT_SHADER

#version 330
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

uniform sampler2D deferred_diffuse;
uniform sampler2D deferred_normal;
uniform sampler2D deferred_depth;
uniform sampler2D deferred_position;
uniform vec2 screen_size;

uniform vec4 color;
uniform vec3 attenuation;
uniform vec4 position;

in vec3 vertexMV;
in vec3 vertex;
in vec3 lightPos;

layout(location=0) out vec3 out_color;


void main() {



//    out_color =   CalcPointLight(vec3(diffColor),position, normal);
    out_color = vec3(color);
}


