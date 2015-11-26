
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

uniform vec4 position;


out vec3 vertexMV;
out vec3 vertex;
out vec3 lightPos;

void main() {
    lightPos = vec3(view  * vec4(model[3]));
    vertexMV = vec3(view * model * vec4( in_position, 1 ));
    vertex = vec3(model * vec4( in_position, 1 ));
    gl_Position = proj*view *model* vec4(in_position,1);
}





##GL_FRAGMENT_SHADER
#version 400

#ifdef SHADOWS
uniform samplerCubeShadow depthTex;
#endif

uniform vec3 attenuation;
uniform vec4 position;
uniform vec2 shadowPlanes; //near and far plane for shadow mapping camera

in vec3 vertexMV;
in vec3 vertex;
in vec3 lightPos;

#include "lighting_helper_fs.glsl"


layout(location=0) out vec4 out_color;


void main() {
    vec3 diffColor,vposition,normal,data;
    float depth;
    getGbufferData(diffColor,vposition,depth,normal,data);
    vec3 specColor = vec3(1);
    vec3 lightDir = normalize(lightPos-vposition);

    float intensity = color.w;
    vec3 lightColor = color.rgb;




    float visibility = 1.0f;
#ifdef SHADOWS

    float farplane = shadowPlanes.x;
    float nearplane = shadowPlanes.y;
    vec3 lightW = vec3(model[3]);
    vec3 fragW = vec3(inverse(view)*vec4(vposition,1));
    visibility = calculateShadowCube(depthTex,lightW,fragW,farplane,nearplane);
#endif

    float atten = getAttenuation(attenuation,distance(vposition,lightPos),position.w);
    float localIntensity = intensity*atten*visibility; //amount of light reaching the given point


    float Idiff = localIntensity * intensityDiffuse(normal,lightDir);
    float Ispec = localIntensity * intensitySpecular(vposition,normal,lightDir,40);


//    out_color = vec4(lightColor*( Idiff*diffColor + Ispec*specColor),1);
    out_color = vec4(lightColor*Idiff ,Ispec); //accumulation


}


