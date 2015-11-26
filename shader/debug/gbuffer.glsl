
##GL_VERTEX_SHADER

#version 400

layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec2 in_tex;

uniform mat4 model;
uniform mat4 proj;


out vec2 texCoord;

void main() {
    texCoord = in_tex;
//    gl_Position = proj * model * vec4(in_position,1);
    gl_Position = model * vec4(in_position.x,in_position.y,0,1);
//    gl_Position = vec4(in_position.x,in_position.y,0,1);
}





##GL_FRAGMENT_SHADER

#version 400
uniform mat4 model;
uniform mat4 proj;

uniform sampler2D text;


in vec2 texCoord;

out vec4 out_color;

void main() {
    vec4 diffColor = texture(text,texCoord);
    out_color =  diffColor;
}


