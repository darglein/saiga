/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/geometry/raytracer.h"
#include "internal/noGraphicsAPI.h"
namespace Saiga {

Raytracer::Result Raytracer::trace(Ray &r){
    Result res;
    res.distance = std::numeric_limits<float>::infinity();

    for(unsigned int i =0;i<triangles.size();++i){
        float d;
        bool back;
        if(r.intersectTriangle(triangles[i],d,back)){
//            std::cout<<"found "<<d<<" back "<<back<<std::endl;
            if( d<res.distance){
                res.distance = d;
                res.triangle = i;
                res.back = back;
            }
        }
    }

    res.valid = res.distance != std::numeric_limits<float>::infinity();
    return res;
}


int Raytracer::trace(Ray &r, std::vector<Result> &output){
    int count = 0;

    for(unsigned int i =0;i<triangles.size();++i){
        float d;
        bool back;
        if(r.intersectTriangle(triangles[i],d,back)){
            Result res;
            res.valid = true;
            res.distance = d;
            res.triangle = i;
            res.back = back;
            output.push_back(res);
            count++;
        }
    }

    return count;
}

}
