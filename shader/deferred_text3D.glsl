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


out vec2 texCoord;

void main() {
    texCoord = in_tex;
//    gl_Position = proj * model * vec4(in_position,1);

//    vec4 p =  model * vec4(in_position,1);

//   mat4 m =  mat4(1.0);
//   m[3] = view[3];
//    gl_Position = proj * m * p;

    gl_Position = proj * view * model * vec4(in_position,1);
}


##end

##start
##fragment

#version 150
#extension GL_ARB_explicit_attrib_location : enable
uniform mat4 model;
uniform mat4 proj;

uniform vec4 color;
uniform vec4 strokeColor;

uniform sampler2D text;


in vec2 texCoord;

out vec4 out_color;

void main() {

    vec2 data = texture(text,texCoord).rg;

    float stroke = data.x;
    float fill = data.y;
    stroke = stroke * (1.0f-fill);


    out_color =  color*fill + strokeColor*stroke;
}

##end
