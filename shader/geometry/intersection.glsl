/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


bool RaySphere(vec3 rayOrigin, vec3 rayDir, vec3 spherePos, float sphereRadius, out float t1, out float t2)
{
    vec3 L = rayOrigin - spherePos;
    float a = dot(rayDir,rayDir);
    float b = 2*dot(rayDir,L);
    float c = dot(L,L) - sphereRadius * sphereRadius;
    float D = b*b + (-4.0f)*a*c;

    // rays misses sphere
    if (D < 0)
        return false;


    if(D==0){
        //ray touches sphere
        t1 = t2 = - 0.5 * b / a;
    }else{
        //ray interescts sphere
        t1 = -0.5 * (b + sqrt(D)) / a ;
        t2 =  -0.5 * (b - sqrt(D)) / a;
    }

    if (t1 > t2){
        float tmp = t1;
        t1 = t2;
        t2 = tmp;
//        std::swap(t1, t2);
    }
    return true;
}
