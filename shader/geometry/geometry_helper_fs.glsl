/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

layout(location=0) out vec3 out_color;
layout(location=1) out vec3 out_normal;
layout(location=2) out vec4 out_data;
layout(location=3) out vec3 out_position;


vec3 packNormal(vec3 normal){
    return normal *0.5f+vec3(0.5f);
}


//not working
vec2 packNormal2 (vec3 n)
{
    vec2 enc = normalize(n.xy) * sqrt(n.z*0.5+0.5);
//    enc = enc*0.5+vec2(0.5);
//    return enc;

    return vec2(sqrt(n.z*0.5+0.5));
}


//Lambert azimuthal equal-area projection
//http://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection
vec2 packNormal3 (vec3 n)
{
    //this projection is undefined at (0,0,-1)
    //that doesn't matter though because the normals are in view space and (0,0,-1) would mean,
    //that the normal points away from the camera
    float f = sqrt(8*n.z+8);
    return n.xy / f + 0.5;
}


//void setGbufferData(vec3 color, vec3 position, vec3 normal, vec3 data){

//    out_color = color;
//    out_position = position;
//    out_normal = packNormal(normalize(normal));
//    out_data = data;

//}


void setGbufferData(vec3 color, vec3 normal, vec4 data){
    out_color = color;
//    out_position = position;
    out_normal.xy = packNormal3(normalize(normal));
//    out_normal = packNormal(normalize(normal));
    out_data = data;
}
