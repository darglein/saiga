/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec3 in_position;


out vec2 tc;


void main() {
    tc = vec2(in_position.x,in_position.y);
    tc = tc*0.5f+0.5f;
    gl_Position = vec4(in_position.x,in_position.y,0,1);
}





##GL_FRAGMENT_SHADER

#version 330


uniform sampler2D image;

uniform vec4 screenSize;
uniform vec4 ssrData; //vec4(stride,jitter,zThickness,maxSteps)
uniform float useBinarySearch;

in vec2 tc;

#include "../lighting/lighting_helper_fs.glsl"

layout(location=0) out vec3 out_color;

float reconstructCSZ(float d) {
    //hard coded near and far plane :)
    float z_n = 0.1f;
    float z_f = 100.0f;

    vec3 clipInfo = vec3(z_n * z_f, z_n - z_f, z_f);

    return -clipInfo[0] / (clipInfo[1] * d + clipInfo[2]);
}


float distanceSquared(vec2 p1, vec2 p2){
    //very efficient...
    float d = distance(p1,p2);
    return d*d;
}

void swap(in out float f1, in out float f2){
    float tmp = f1;
    f1 = f2;
    f2 = tmp;
}

vec2 toScreenSpace(vec3 csPos){
     vec4 h = proj * vec4(csPos, 1.0);
     float k0 = 1.0 / h.w;
     h = h * k0;
     h = h*0.5f + 0.5f;
     return h.xy * screen_size;
}

vec3 traceScreenSpaceRay(vec3 csOrig, vec3 csDir, mat4 proj,
                         sampler2D csZBuffer, vec2 csZBufferSize, float zThickness,
     float nearPlaneZ, float stride, float jitter, const float maxSteps, float maxDistance, bool useBinarySearch,
    out vec2 hitPixel, out vec3 csHitPoint){


    // Clip to the near plane
    float rayLength = ((csOrig.z + csDir.z * maxDistance) > -nearPlaneZ) ? (nearPlaneZ + csOrig.z) / -csDir.z : maxDistance;

    vec3 csEndPoint = csOrig + csDir * rayLength;
    hitPixel = vec2(-1, -1);



    // Project into screen space
    vec4 H0 = proj * vec4(csOrig, 1.0), H1 = proj * vec4(csEndPoint, 1.0);
    float k0 = 1.0 / H0.w, k1 = 1.0 / H1.w;
    vec3 Q0 = csOrig * k0, Q1 = csEndPoint * k1;
    // Screen-space endpoints
    vec2 P0 = H0.xy * k0;
    vec2 P1 = H1.xy * k1;


    P0 = toScreenSpace(csOrig);
    P1 = toScreenSpace(csEndPoint);


    float xMax=csZBufferSize.x-0.5, xMin=0.5, yMax=csZBufferSize.y-0.5, yMin=0.5;
    float alpha = 0.0;
    // Assume P0 is in the viewport (P1 - P0 is never zero when clipping)
    if ((P1.y > yMax) || (P1.y < yMin))
    alpha = (P1.y - ((P1.y > yMax) ? yMax : yMin)) / (P1.y - P0.y);
    if ((P1.x > xMax) || (P1.x < xMin))
    alpha = max(alpha, (P1.x - ((P1.x > xMax) ? xMax : xMin)) / (P1.x - P0.x));
    P1 = mix(P1, P0, alpha);
    k1 = mix(k1, k0, alpha);
    Q1 = mix(Q1, Q0, alpha);


    P1 += vec2((distanceSquared(P0, P1) < 0.0001) ? 0.01 : 0.0);


    vec2 delta = P1 - P0;
    bool permute = false;
    if (abs(delta.x) < abs(delta.y)) {
        permute = true;
        delta = delta.yx; P0 = P0.yx; P1 = P1.yx;
    }

    float stepDir = sign(delta.x), invdx = stepDir / delta.x;

    // Track the derivatives of Q and k.
    vec3 dQ = (Q1 - Q0) * invdx;
    float dk = (k1 - k0) * invdx;
    vec2 dP = vec2(stepDir, delta.y * invdx);

    dP *= stride; dQ *= stride; dk *= stride;
    P0 += dP * jitter; Q0 += dQ * jitter; k0 += dk * jitter;
    float prevZMaxEstimate = csOrig.z;

    // Slide P from P0 to P1, (now-homogeneous) Q from Q0 to Q1, k from k0 to k1
    vec3 Q = Q0; float k = k0, stepCount = 0.0, end = P1.x * stepDir;
    int binarySearches = useBinarySearch ? int(round(log2(stride))) : 1;
//    binarySearches = 3;
    bool inBinarySearch = false;
    vec2 bestHit;

    for (vec2 P = P0;
        ((P.x * stepDir) <= end) && (stepCount < maxSteps);
        P += dP, Q.z += dQ.z, k += dk, stepCount += 1.0) {

        // Project back from homogeneous to camera space
        hitPixel = permute ? P.yx : P;

        // The depth range that the ray covers within this loop iteration.
        // Assume that the ray is moving in increasing z and swap if backwards.
        float rayZMin = prevZMaxEstimate;
        float rayZMax = ( Q.z) / ( k);
        prevZMaxEstimate = rayZMax;
        if (rayZMin > rayZMax) {
            swap(rayZMin, rayZMax);
        }

        // Camera-space z of the background
        float sceneZMax = texelFetch(csZBuffer, ivec2(hitPixel), 0).r;
        sceneZMax = reconstructCSZ(sceneZMax);
        float sceneZMin = sceneZMax - zThickness;

        if (((rayZMax >= sceneZMin) && (rayZMin <= sceneZMax)) ||
            (sceneZMax == 0)) {
                //hit point found
                bestHit = hitPixel;

                //go one step back and half the stepsize
                P -= dP;Q.z -= dQ.z; k -= dk;
                dP *= 0.5f; dQ *= 0.5f; dk *=0.5f;
                prevZMaxEstimate = (Q.z) / ( k);


                //start binary search
                binarySearches--;
                inBinarySearch = true;



        }else if(inBinarySearch){
            //no hit found -> only half the stepsize
             dP *= 0.5f; dQ *= 0.5f; dk *=0.5f;
             binarySearches--;
        }

        if(binarySearches==0){
            //done
            return texelFetch( image, ivec2(bestHit),0 ).rgb;
        }
    }


    // Advance Q based on the number of steps
    Q.xy += dQ.xy * stepCount;
    vec3 hitPoint = Q * (1.0 / k);
    hitPixel = hitPoint.xy;
    return vec3(1,0,0);
}



void main() {

    vec3 diffColor,position,normal,data;
    float depth;
    getGbufferData(diffColor,position,depth,normal,data);



    vec3 color = texture( image, tc ).rgb;
    if(data.z==0){
        out_color = color;
        return;
    }

    vec3 csOrig = position;
    vec3 csDir = normalize(position);
    csDir = normalize(reflect(csDir,normal));

    vec3 clipInfo;
    vec2 hitPixel;
    vec3 csHitPoint;

    float stride = 4.0f;
    float jitter = 1.0f;
    float zThickness = 2.0f;
    float maxSteps = 400;

    stride = ssrData.x;
    jitter = ssrData.y;
    zThickness = ssrData.z;
    maxSteps = ssrData.w;

    bool ubs = useBinarySearch==1.0f;

    float maxDistance = 50.0f;

      vec3 rcolor =  traceScreenSpaceRay(csOrig,csDir,proj,deferred_depth,screen_size,zThickness,0.1f,stride,jitter,maxSteps,maxDistance,ubs,hitPixel,csHitPoint);

      out_color = mix(color,rcolor,data.z);
      return;




}


