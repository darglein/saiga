
##GL_VERTEX_SHADER

#version 400
layout(location=0) in vec3 in_position;

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


#ifdef SHADOWS
uniform sampler2DShadow depthTex;
#endif

uniform sampler2D ssaoTex;

uniform vec3 direction;
uniform float ambientIntensity;

#include "lighting_helper_fs.glsl"

layout(location=0) out vec4 out_color;


float getSSAOIntensity(){
//    ivec2 tci = ivec2(gl_FragCoord.xy);
//    float ssao = texelFetch(ssaoTex,tci,0).r;
//    return ssao;
    return 1.0f;
}

vec4 getDirectionalLightIntensity(int sampleId) {
    vec3 diffColor,vposition,normal,data;
    float depth;
    getGbufferData(diffColor,vposition,depth,normal,data,sampleId);

    vec3 lightDir = direction;
    float ssao = getSSAOIntensity();
    float intensity = lightColorDiffuse.w;

    float visibility = 1.0f;
#ifdef SHADOWS
    visibility = calculateShadow(depthTex,vposition,1.0f);
//    visibility = calculateShadowPCF(depthTex,vposition,1.0f);
#endif

    float localIntensity = intensity * visibility; //amount of light reaching the given point

    float Iamb = intensity * ambientIntensity * ssao;
    float Idiff = localIntensity * intensityDiffuse(normal,lightDir);
    float Ispec = 0;
    if(Idiff > 0)
        Ispec = localIntensity * data.x  * intensitySpecular(vposition,normal,lightDir,40);

    float Iemissive = data.y ;

    vec3 color = lightColorDiffuse.rgb * (
                Idiff * diffColor +
                Ispec * lightColorSpecular.w * lightColorSpecular.rgb +
                Iamb * diffColor) +
            Iemissive * diffColor;

//    return vec4(1);
    return vec4(color,1);
//    return vec4(lightColor*(Idiff+Iamb) ,Ispec); //accumulation
//    out_color = vec4(1.0f);
}

void main(){
//    vec4 c = vec4(0);
//    int s = 1;
//    for(int i = 0 ; i < s ; ++i){
//        c += getDirectionalLightIntensity(i);
//    }
//    out_color = (c / s);

    out_color = getDirectionalLightIntensity(0);
}


