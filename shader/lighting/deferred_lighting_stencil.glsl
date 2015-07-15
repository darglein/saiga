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



//layout(location=0) out vec3 out_color;
//layout(location=3) out vec3 out_color2;
void main() {
//    out_color = vec3(0,1,0);
//    out_color2 = vec3(0,1,0);
}

##end
