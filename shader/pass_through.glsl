##start
##vertex

#version 150
#extension GL_ARB_explicit_attrib_location : enable
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec2 in_tex;

out vec2 texCoord;

void main() {
    texCoord = in_tex;
    gl_Position = vec4(in_position,1);
}


##end

##start
##fragment

#version 150

in vec2 texCoord;

uniform sampler2D text;

out vec4 color;



void main(){
    color = texture( text, texCoord );
}

##end
