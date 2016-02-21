
##GL_VERTEX_SHADER

#version 400
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec2 in_tex;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

uniform mat4 MV;
uniform mat4 MVP;


out vec2 texCoord;

void main() {
//    gl_Position = vec4( in_position, 1 );
    texCoord = in_tex;
    gl_Position = vec4(in_position.x,in_position.y,1,1);
}





##GL_FRAGMENT_SHADER

#version 400
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform sampler2D image;


in vec2 texCoord;

layout(location=0) out vec4 out_color;

void main() {
     float d = texture(image, texCoord).r;
    gl_FragDepth = d;
    out_color = vec4(0);
}


