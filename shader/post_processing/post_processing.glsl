
##GL_VERTEX_SHADER

#version 400
layout(location=0) in vec3 in_position;


out vec2 tc;


void main() {
    tc = vec2(in_position.x,in_position.y);
    tc = tc*0.5f+0.5f;
    gl_Position = vec4(in_position.x,in_position.y,0,1);
}





##GL_FRAGMENT_SHADER

#version 400


uniform sampler2D image;

uniform vec4 screenSize;

in vec2 tc;

layout(location=0) out vec3 out_color;




void main() {

    //load data from gbuffer
//    vec2 tc = CalcTexCoord();
//    vec4 diffColor = texture( deferred_diffuse, tc );
    out_color = texture( image, tc ).rgb;
//    out_color = vec3(0,1,0);


}


