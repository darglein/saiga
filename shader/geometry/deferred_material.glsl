/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec2 in_tex;

#include "camera.glsl"
uniform mat4 model;

out vec3 normal;
out vec3 normalW;
out vec3 vertexMV;
out vec3 vertex;
out vec2 texCoord;

void main() {
//    gl_Position = vec4( in_position, 1 );
    texCoord = in_tex;
    normal = normalize(vec3(MV * vec4( in_normal, 0 )));
    normalW = normalize(vec3(model * vec4( in_normal, 0 )));
    vertexMV = vec3(view * model * vec4( in_position, 1 ));
    vertex = vec3(model * vec4( in_position, 1 ));
    gl_Position = MVP * vec4(in_position,1);
}





##GL_FRAGMENT_SHADER

#version 330


uniform vec3 colors[3];
uniform sampler2D textures[5];
uniform float use_textures[5];

in vec3 normal;
in vec3 normalW;
in vec3 vertexMV;
in vec3 vertex;
in vec2 texCoord;

#include "geometry_helper_fs.glsl"

// http://www.thetenthplanet.de/archives/1180
mat3 cotangent_frame(vec3 N, vec3 p, vec2 uv)
{
    // get edge vectors of the pixel triangle
    vec3 dp1 = dFdx( p );
    vec3 dp2 = dFdy( p );
    vec2 duv1 = dFdx( uv );
    vec2 duv2 = dFdy( uv );

    // solve the linear system
    vec3 dp2perp = cross( dp2, N );
    vec3 dp1perp = cross( N, dp1 );
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;

    // construct a scale-invariant frame
    float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
    return mat3( T * invmax, B * invmax, N );
}

vec3 perturb_normal( vec3 N, vec3 V, vec2 texcoord )
{
    // assume N, the interpolated vertex normal and
    // V, the view vector (vertex to eye)
   vec3 map = texture(textures[4], texcoord ).xyz;
   map = map * 255./127. - 128./127.;
    mat3 TBN = cotangent_frame(N, -V, texcoord);
    return normalize(TBN * map);
}


void main() {

    vec4 alpha = texture(textures[3], texCoord)*use_textures[3] + (vec4(1)-use_textures[3])*vec4(1,1,1,1);
    if(alpha.x==0)
        discard;
    vec4 ambColor = texture(textures[0], texCoord)*use_textures[0] + (vec4(1)-use_textures[0])*vec4(colors[0],1);
    vec4 diffColor = texture(textures[1], texCoord)*use_textures[1] + (vec4(1)-use_textures[1])*vec4(colors[1],1);
    vec4 specColor = texture(textures[2], texCoord)*use_textures[2] + (vec4(1)-use_textures[2])*vec4(colors[2],1);


    vec3 E = normalize(-vertexMV);
    vec3 N = normalize(normal);

    //normal mapping
    if(use_textures[4]>0)
        N = perturb_normal(N,E,texCoord);



    // write Total Color:
//    out_color =  vec3(diffColor);
////    out_normal = vec3(normalW);
//    out_normal = vec3(N)*0.5f+0.5f;
//    out_position = vertexMV;
//    LFragment.w = 1;
//    LFragment = vec4( normal, 1.0 );
//    float spec = (specColor.r+specColor.g+specColor.b)*0.333f;
    float spec = 0.3f;
    setGbufferData(vec3(diffColor),N,vec3(spec,0,0));
}


