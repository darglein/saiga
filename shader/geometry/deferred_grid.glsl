##start
##vertex

#version 400
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

uniform mat4 MV;
uniform mat4 MVP;

out vec3 normal;
out vec3 vertexMV;
out vec3 vertex;

void main() {
    normal = normalize(vec3(view*model * vec4( in_normal, 0 )));
    vertexMV = vec3(view * model * vec4( in_position, 1 ));
    vertex = vec3(model * vec4( in_position, 1 ));
    gl_Position = proj*view *model* vec4(in_position,1);
}


##end

##start
##fragment

#version 400
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform vec4 color;

in vec3 normal;
in vec3 vertexMV;
in vec3 vertex;

layout(location=0) out vec3 out_color;
layout(location=1) out vec3 out_normal;
layout(location=2) out vec3 out_position;

void main() {

    vec4 diffColor = color;

    out_color =  vec3(diffColor);
    out_normal = normalize(normal)*0.5f+0.5f;
//    out_normal = vec3(0,1,0);
    out_position = vertexMV;
}

##end
