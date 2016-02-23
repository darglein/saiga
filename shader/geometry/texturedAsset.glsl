
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

out vec3 normal;
out vec3 normalW;
out vec3 vertexMV;
out vec3 vertex;
out vec2 texCoord;

void main() {
//    gl_Position = vec4( in_position, 1 );
    texCoord = in_tex;
    normal = normalize(vec3(view*model * vec4( in_normal, 0 )));
    normalW = normalize(vec3(model * vec4( in_normal, 0 )));
    vertexMV = vec3(view * model * vec4( in_position, 1 ));
    vertex = vec3(model * vec4( in_position, 1 ));
    gl_Position = proj*view *model* vec4(in_position,1);
}





##GL_FRAGMENT_SHADER

#version 400
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform sampler2D image;
uniform float userData; //blue channel of data texture in gbuffer. Not used in lighting.

in vec3 normal;
in vec3 normalW;
in vec3 vertexMV;
in vec3 vertex;
in vec2 texCoord;



#include "geometry_helper_fs.glsl"


void main() {
    vec4 diffColor = texture(image, texCoord);
    vec3 data = vec3(1,0,0);
    setGbufferData(vec3(diffColor),vertexMV,normal,vec4(data.xy,userData,0));
}


