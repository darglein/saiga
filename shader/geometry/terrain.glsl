/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec3 in_position;

#include "camera.glsl"
uniform mat4 model;

//current level
uniform sampler2D image;
uniform sampler2D normalMap;

//coarser level
uniform sampler2D imageUp;
uniform sampler2D normalMapUp;

uniform vec4 ScaleFactor, FineBlockOrig, TexSizeScale;

uniform vec2 RingSize, ViewerPos, AlphaOffset, OneOverWidth;
uniform float ZScaleFactor, ZTexScaleFactor;

//out vec3 normal;
out vec3 vertexMV;
out vec2 otc;
out float alpha;
out vec3 vertex;
out vec3 normal2;



void main() {

    // convert from grid xy to world xy coordinates
     //  ScaleFactor.xy: grid spacing of current level
     //  ScaleFactor.zw: origin of current block within world
    vec2 worldPos = in_position.xz * ScaleFactor.xy + ScaleFactor.zw;


    vec4 position = vec4(worldPos.x+ViewerPos.x,0,worldPos.y+ViewerPos.y,1);


    vec2 a = abs(worldPos)/(RingSize*0.5f);

    //TODO:: Better alpha
    alpha = clamp(max(a.x,a.y),0,1);
    alpha = clamp(alpha - 0.7f,0,1);
    alpha = alpha * (1.0/0.2f);
    alpha = clamp(alpha,0,1);
//    alpha = 0;

//    vec2 textureSize = vec2(5000,5000);
//    tc = vec2(position.xz)/textureSize;
    vec2 tc = (vec2(position.xz)+TexSizeScale.xy)*TexSizeScale.zw;
//    tc = tc * TexSizeScale.zw;
//    tc = (vec2(position.xz)+0.5f*vec2(0.01f))/vec2(0.01f);
    otc = tc;
//    tc = (in_position.xz+0.5f)*FineBlockOrig.xy+FineBlockOrig.zw;


    // sample the vertex texture
//    float height = texture(normalMap,tc).r;
    float height1 = texture(image,tc).r;
    height1 = height1*ZScaleFactor;

    float height2 = texture(imageUp,tc).r;
    height2 = height2*ZScaleFactor;


    position.y = (1-alpha)*height1+alpha*height2;

    vertex = vec3(position);
    vertexMV = vec3(view * position);
    gl_Position = viewProj * position;

//    c = texture(image,tc).rrr;

}







##GL_FRAGMENT_SHADER

#version 330
#include "camera.glsl"
uniform mat4 model;
uniform vec4 color;
uniform sampler2D normalMap;

uniform sampler2D texture1;
uniform sampler2D texture2;
uniform sampler2D image;

uniform float ZScaleFactor;

//in vec3 normal;
in vec3 vertexMV;
in vec2 otc;
in float alpha;
in vec3 vertex;
in vec3 normal2;

layout(location=0) out vec3 out_color;
layout(location=1) out vec3 out_normal;
layout(location=2) out vec3 out_position;

vec4 triplanarTexturing(vec3 normal){

    vec3 plateauSize = vec3(0);
    vec3 transitionSpeed = vec3(1.5);
    vec3 texScale = vec3(0.01f,1.0f/ZScaleFactor,0.01f);

    vec3 blendWeights = abs(normal);
    blendWeights = (blendWeights - 0.4) * 7;
    blendWeights = max(blendWeights, 0);      // Force weights to sum to 1.0 (very important!)
    float sum = (blendWeights.x + blendWeights.y + blendWeights.z );
    blendWeights /= sum;

//    blendWeights = pow(max(blendWeights, 0), transitionSpeed);
    vec2 coord1 = (vertex.yz ) * texScale.yz;
    vec2 coord2 = (vertex.zx ) * texScale.zx;
    vec2 coord3 = (vertex.xy ) * texScale.xy;

    vec3 col1 = texture(texture2, coord1).xyz * blendWeights.x;

    vec3 col3 = texture(texture2, coord3).xyz * blendWeights.z;



    vec3  col21 = texture(texture2, coord2).xyz * blendWeights.y;
    vec3  col22 = texture(texture1, coord2).xyz * blendWeights.y;

    float factor = vertex.y*1.0f/ZScaleFactor;
    factor = (factor-0.1f) * 2.0f;
    factor = clamp(factor,0,1);

    vec3 col2 = factor*col22 + (1.0-factor)*col21;
//    vec4 textColour = vec4(col1.xyz * blendWeights.x +
//        col2.xyz * blendWeights.y +
//        col3.xyz * blendWeights.z, 1);

    vec4 textColour = vec4(col1+col2+col3,1);
//    vec4 textColour = vec4(col2+vec3(blendWeights.x,0,blendWeights.z),1);



    return textColour;
}



void main() {

    vec3 col1 = texture(texture1,7 * otc).rgb;
    vec3 col2 = texture(texture2,5 * otc).rgb;

    float blend_alpha = vertex.y / ZScaleFactor * 1.3;
    blend_alpha = clamp(blend_alpha, 0, 1);
    vec3 col = (1-blend_alpha) * col1 + blend_alpha * col2;

    vec3 n = texture(normalMap,otc).xyz;
    n = n*2.0f - vec3(1.0f);
    n = normalize(n);
    vec3 normal = normalize(vec3(view * vec4(n, 0 )));
/*
   vec4 diffColor = vec4(color);
   // vec4 diffColor = vec4(alpha);

   if(alpha>=0.95f){
       diffColor = vec4(1,0,0,1);
   }else{
       diffColor = vec4(1);
   }
//    out_color =  vec3(triplanarTexturing(normal));
   // out_color =  texture(texture1,7 * otc).rgb;
    // out_color = vec3(1);
*/
    out_color =  vec3(col);
     out_color =  vec3(triplanarTexturing(n));
    out_normal = normal*0.5f+vec3(0.5f);
    out_position = vertexMV;
}


