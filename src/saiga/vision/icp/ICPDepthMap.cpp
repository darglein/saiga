/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "ICPDepthMap.h"
#include "saiga/time/timer.h"

namespace Saiga {

using namespace Depthmap;

namespace ICP {



std::vector<Correspondence> projectiveCorrespondences(DepthPointCloud refPc, DepthNormalMap refNormal,
                                                      DepthPointCloud srcPc, DepthNormalMap srcNormal,
                                                      SE3 T, Intrinsics4 camera,
                                                      double distanceThres, double cosNormalThres
                                                      , int searchRadius, bool useInvDepthAsWeight, bool scaleDistanceThresByDepth)
{
    std::vector<Correspondence> result;
    result.reserve(srcPc.h * srcPc.w);

    for(int i = 0; i < srcPc.h; ++i)
    {
        for(int j = 0; j < srcPc.w; ++j)
        {
            Vec3 p0 = srcPc(i,j);
            Vec3 n0 = srcNormal(i,j);

            Vec3 p = p0;
            Vec3 n = n0;

            if(!p.allFinite() || !n.allFinite())
                continue;

            // transform point and normal to reference frame
            p = T * p;
            n = T.so3() * n;

            // project point to reference to find correspondences
            Vec2 ip = camera.project(p);

            // round to nearest integer
            ip = ip.array().round();

            int sx = ip(0);
            int sy = ip(1);




            double bestDist = std::numeric_limits<double>::infinity();
            Correspondence corr;

            // search in a small neighbourhood of the projection
            int S = searchRadius;
            for(int dy = -S; dy <= S; ++dy)
            {
                for(int dx = -S; dx <= S; ++dx)
                {
                    int x = sx + dx;
                    int y = sy + dy;

                    if(!refPc.inImage(y,x))
                        continue;

                    Vec3 p2 = refPc(y,x);
                    Vec3 n2 = refNormal(y,x);


                    if(!p2.allFinite() || !n2.allFinite())
                        continue;

                    auto distance = (p2-p).norm();

                    auto depth = p2(2);
                    auto invDepth = 1.0 / depth;

                    auto disTh = scaleDistanceThresByDepth ? distanceThres * depth : distanceThres;

                    if( distance < bestDist && distance < disTh && n.dot(n2) > cosNormalThres )
                    {
                        corr.refPoint = p2;
                        corr.refNormal = n2;
                        corr.srcPoint = p0;
                        corr.srcNormal = n0;
                        corr.weight = useInvDepthAsWeight ? invDepth * invDepth : 1;
                        bestDist = distance;
                    }
                }
            }

            if(std::isfinite(bestDist))
            {
                result.push_back(corr);
            }

        }
    }

    return result;
}

SE3 alignDepthMaps(DepthMap referenceDepthMap, DepthMap sourceDepthMap, SE3 guess, Intrinsics4 camera, int iterations)
{
//    SAIGA_BLOCK_TIMER;

    Saiga::ArrayImage<Vec3> refPc(referenceDepthMap.h, referenceDepthMap.w);
    Saiga::ArrayImage<Vec3> srcPc(sourceDepthMap.h, sourceDepthMap.w);

    Saiga::ArrayImage<Vec3> refNormal(referenceDepthMap.h, referenceDepthMap.w);
    Saiga::ArrayImage<Vec3> srcNormal(sourceDepthMap.h, sourceDepthMap.w);

    Saiga::Depthmap::toPointCloud(referenceDepthMap,refPc,camera);
    Saiga::Depthmap::toPointCloud(sourceDepthMap,srcPc,camera);


    Saiga::Depthmap::normalMap(refPc,refNormal);
    Saiga::Depthmap::normalMap(srcPc,srcNormal);

    SE3 T = guess;

    std::vector<Saiga::ICP::Correspondence> corrs;

    for(int k = 0; k < iterations; ++k)
    {
        {
//            SAIGA_BLOCK_TIMER;
            corrs = Saiga::ICP::projectiveCorrespondences(refPc,refNormal,srcPc,srcNormal,T,camera);
        }
        {
//            SAIGA_BLOCK_TIMER;
            T = Saiga::ICP::pointToPlane(corrs,T);
        }
    }
    return T;
}

}
}
