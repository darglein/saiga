##start
##vertex

#version 150
#extension GL_ARB_explicit_attrib_location : enable
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


##end

##start
##fragment

#version 150
#extension GL_ARB_explicit_attrib_location : enable
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform sampler2D image;
uniform float highlight;

in vec3 normal;
in vec3 normalW;
in vec3 vertexMV;
in vec3 vertex;
in vec2 texCoord;

layout(location=0) out vec3 out_color;
layout(location=1) out vec3 out_normal;
layout(location=2) out vec3 out_position;

vec3 blackAndWhite(vec3 color){
//    float light = dot(color,vec3(1))/3.0f;
    float light = dot(color,vec3(0.21f,0.72f,0.07f));
    return vec3(light);
}

void main() {

     vec3 diffColor = texture(image, texCoord).rgb;

    out_color =  highlight*diffColor + (1.0f-highlight) * blackAndWhite(diffColor);
    out_normal = normalize(normal)*0.5f+0.5f;
    out_position = vertexMV;
}

##end
