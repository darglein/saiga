/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec3 in_position;
layout(location=1) in vec4 in_velocity;
layout(location=2) in vec3 in_force;
layout(location=3) in vec3 in_right;
layout(location=4) in vec4 in_scale;
layout(location=5) in vec4 data;
layout(location=6) in float in_startFade;
layout(location=7) in ivec3 idata;

#include "camera.glsl"
uniform mat4 model;

uniform int timer;
uniform float timestep;
uniform float interpolation;

out vec2 scale;

out vec3 direction;
out vec3 right;

out float specular;
out float fade;
flat out int layer;
flat out int orientation;

float getTime(){
    int time = idata.x;
    float t = float(timer-time)*timestep;
    t = t+(interpolation*timestep);
    return t;
}

void setBasicOutputs(float t){
    float fadetime = data.z;
    float lifetime = data.y;

    if(t>=lifetime)
        fade = 0;
    else
        fade = clamp(1-((t-fadetime)/(lifetime-fadetime)),0,1) * in_startFade;

    layer = idata.y;
    orientation = idata.z;


    scale = in_scale.xy + in_scale.zw * t;
    right = in_right;
    specular = data.w;
}

void main() {
    float t = getTime();
    setBasicOutputs(t);

    float drag = data.x;
    vec3 force = in_force;
//    force -= 3.0f * in_velocity.xyz * in_velocity.w;

    //exact integration. (no approximation, because 't' is the total time and not a timestep)
    vec3 v0 = in_velocity.xyz * in_velocity.w ;
    vec3 velocity = v0*exp(-t*drag)+ force * t;
    vec3 position = in_position.xyz + velocity * t - 0.5f * t * t * force;

//    position = in_position.xyz + v0 * t + v0*(1.0f-exp(-t));


    direction = normalize(vec3(model*vec4(in_velocity.xyz + in_force * t,0)));
    gl_Position = model* vec4(position,1);

}






##GL_GEOMETRY_SHADER
#version 330

layout(points) in;
layout(triangle_strip, max_vertices=4) out;

//in vec3[1] direction;
//in vec3[1] right;
//in vec2[1] scale;

//in float[1] fade;
//in float[1] specular;
//flat in int[1] layer;
//flat in int[1] orientation;

in vec3 direction[];
in vec3 right[];
in vec2 scale[];

in float fade[];
in float specular[];
flat in int layer[];
flat in int orientation[];


#include "camera.glsl"
uniform mat4 model;

out vec3 tc;
out float fade2;
out float specular2;
out vec3 normal;
out vec3 vertexMV;

vec3 getTC(int id){
     vec4 tx=vec4(0,1,0,1);
     vec4 ty=vec4(0,0,1,1);
    return vec3(tx[id],ty[id],layer[0]);
}

vec4 getOffset(int id){
     vec4 ix=vec4(-1,1,-1,1);
     vec4 iy=vec4(-1,-1,1,1);

    vec4 pos;
    pos.x =ix[id]*scale[0].x;
    pos.y =iy[id]*scale[0].y;
    pos.z = 0;
    pos.w = 0;
    return pos;
}

vec4 getOffset(int i, vec3 right, vec3 dir){
     vec4 ix=vec4(1,-1,1,-1);
     vec4 iy=vec4(1,1,-1,-1);

    vec3 offset =  ix[i]*dir*scale[0].x+iy[i]*right*scale[0].y;

    return vec4(offset,0);
}



void createVelocityParticle(){
    vec4 position = view*gl_in[0].gl_Position;


    vec3 dir = normalize(vec3(view*vec4(direction[0],0)));

    vec3 right = normalize(cross(dir,vec3(0,0,-1)));

    for(int i =0; i<4;i++){
        vec4 pos = getOffset(i,right,dir)+position;
        tc = getTC(i);
        fade2 = fade[0];
        specular2 = specular[0];
        gl_Position = proj*pos;
        vertexMV = vec3(pos);
        normal = dir;
        EmitVertex();
    }
}

void createBillboardParticle(){
    vec4 position = view*gl_in[0].gl_Position;


    vec3 dir = normalize(cross(right[0],vec3(0,0,-1)));


    for(int i =0; i<4;i++){
        vec4 pos = getOffset(i,dir,right[0])+position;
        tc = getTC(i);
        fade2 = fade[0];
        specular2 = specular[0];
        gl_Position = proj*pos;
        vertexMV = vec3(pos);
        normal = dir;
        EmitVertex();
    }
}


void createFixedParticle(){
    vec4 position = view*gl_in[0].gl_Position;


//    vec3 dir = normalize(vec3(view*vec4(direction[0],0)));

    vec3 dir = direction[0];
//    vec3 right2 = normalize(cross(dir,vec3(0,0,-1)));
    vec3 front = normalize(cross(dir,right[0]));

    for(int i =0; i<4;i++){
        vec4 pos = view*getOffset(i,front,right[0])+position;
        tc = getTC(i);
        fade2 = fade[0];
        specular2 = specular[0];
        gl_Position = proj*pos;
        vertexMV = vec3(pos);
        normal = vec3(view*vec4(dir,0));
        EmitVertex();
    }
}

void main() {
    if(fade[0]==0.0f)
        return;




     if(orientation[0]==0){
         createBillboardParticle();
     }else if(orientation[0]==1){
         createVelocityParticle();
     }else{// if(orientation[0]==2){
         createFixedParticle();
     }



}




##GL_FRAGMENT_SHADER
#version 330

//core in version 4.5 or with extension: ARB_shader_image_load_store

#extension GL_ARB_shader_image_load_store: enable
#ifdef GL_ARB_shader_image_load_store
layout(early_fragment_tests) in; //force early depth tests. may not work on older versions
#endif //GL_ARB_shader_image_load_store


uniform sampler2DArray image;
uniform sampler2D depthTexture;

uniform vec2 cameraParameters;

in vec3 tc;
in float fade2;
in float specular2;
in vec3 normal;
in vec3 vertexMV;

layout(location=0) out vec4 out_color;
//layout(location=1) out vec3 out_normal;
//layout(location=2) out vec3 out_position;
layout(location=1) out vec4 out_data;

float linearDepth(float d){
    float f=cameraParameters.y;
    float n = cameraParameters.x;
    return(2 * n) / (f + n - d * (f - n));
}

float reconstructCSZ(float d) {
    float z_n = cameraParameters.x;
    float z_f = cameraParameters.y;

    vec3 clipInfo = vec3(z_n * z_f, z_n - z_f, z_f);

    return -clipInfo[0] / (clipInfo[1] * d + clipInfo[2]);
}

void main() {

    //these values are scene dependend
    const float maxOffsetL = 0.2f;
    const float maxOffsetH = 0.5f;


    //depth from particle
    float depth = gl_FragCoord.z;
    //depth from background geometry
    float depth2 = texelFetch(depthTexture,ivec2(gl_FragCoord),0).r;


    //world-space-distance between particle and geometry
    float offset = abs(reconstructCSZ(depth) - reconstructCSZ(depth2));
    //fade out particle when background is too far away
    float alpha = 1-smoothstep(maxOffsetL,maxOffsetH,offset);
    //combine distance fade image alpha and particle fade
    vec4 c = texture(image,tc);
    c.a *= alpha * fade2;
    c.a = round(c.a);
    //only write color and data. the normals are not modified
    out_color = c;
    out_data = vec4(specular2,0,0,c.a);
}


