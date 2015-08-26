
##GL_VERTEX_SHADER

#version 150
#extension GL_ARB_explicit_attrib_location : enable
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec2 in_tex;

out vec2 tc;

void main() {
    tc = in_tex;
    gl_Position = vec4(in_position,1);
}





##GL_FRAGMENT_SHADER

#version 150

in vec2 tc;

uniform sampler2D deferred_diffuse;
uniform sampler2D deferred_normal;
uniform sampler2D deferred_depth;
uniform sampler2D deferred_position;

out vec4 color;



void main(){

    vec2 ntc = tc;


    if(tc.x<0.5){
        ntc.x*=2;
        if(tc.y<0.5){
            ntc.y*=2;
            color = texture( deferred_position, ntc);
        }else{
            ntc.y=ntc.y*2-1;
            color = texture( deferred_diffuse, ntc);
        }
    }else{
        ntc.x=ntc.x*2-1;
        if(tc.y<0.5){
            ntc.y*=2;
            color = texture( deferred_depth, ntc);
        }else{
            ntc.y=ntc.y*2-1;
            color = texture( deferred_normal, ntc);
        }
    }

}


