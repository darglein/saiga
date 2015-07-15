##start
##vertex

#version 400
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec2 in_tex;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

uniform mat4 MV;
uniform mat4 MVP;



void main() {
    gl_Position = proj*view *model* vec4(in_position,1);
}


##end

##start
##fragment

#version 400
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;



void main() {


}

##end
