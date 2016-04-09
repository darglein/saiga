
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




void main() {
    gl_Position = vec4(in_position.x,in_position.y,0,1);

}





##GL_FRAGMENT_SHADER
#version 400


uniform sampler2D lightAccumulationtexture;


#include "lighting_helper_fs.glsl"

layout(location=0) out vec4 out_color;

vec4 getAccumulatedColor(int sampleId) {
    vec3 diffColor,vposition,normal,data;
    float depth;
    getGbufferData(diffColor,vposition,depth,normal,data,sampleId);

    ivec2 tci = ivec2(gl_FragCoord.xy);
    vec4 light = texelFetch(lightAccumulationtexture,tci,0);

//    vec4 light = texture(lightAccumulationtexture,CalcTexCoord());

//    vec3 specColor = vec3(1);
    vec3 specColor = light.rgb;

    float specular = data.x;
    float emissive = data.y;

    return vec4(diffColor*(light.rgb+emissive) + specular*specColor*light.w,1); //accumulation
}

void main(){
//    vec4 c = vec4(0);
//    int s = 16;
//    for(int i = 0 ; i < s ; ++i){
//        c += getAccumulatedColor(i);
//    }
//    out_color = (c / s);

//    out_color = getAccumulatedColor(0);




    //pass through
    ivec2 tci = ivec2(gl_FragCoord.xy);
    out_color = texelFetch(lightAccumulationtexture,tci,0);
}



