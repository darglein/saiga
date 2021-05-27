/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "ICPDepthMap.h"

#include "saiga/core/time/timer.h"

namespace Saiga
{
using namespace Depthmap;

namespace ICP
{
DepthMapExtended::DepthMapExtended(const DepthMap& depth, const IntrinsicsPinholed& camera, const SE3& pose)
    : depth(depth), points(depth.h, depth.w), normals(depth.h, depth.w), camera(camera), pose(pose)
{
    Saiga::Depthmap::toPointCloud(depth, points, camera);
    Saiga::Depthmap::normalMap(points, normals);
}

AlignedVector<Correspondence> projectiveCorrespondences(const DepthMapExtended& ref, const DepthMapExtended& src,
                                                        const ProjectiveCorrespondencesParams& params)
{
    AlignedVector<Correspondence> result;
    result.reserve(ref.depth.h * ref.depth.w);


    auto T = ref.pose.inverse() * src.pose;  // A <- B

    for (int i = 0; i < src.depth.h; i += params.stride)
    {
        for (int j = 0; j < src.depth.w; j += params.stride)
        {
            Vec3 p0 = src.points(i, j);
            Vec3 n0 = src.normals(i, j);

            Vec3 p = p0;
            Vec3 n = n0;

            if (!p.allFinite() || !n.allFinite()) continue;

            // transform point and normal to reference frame
            p = T * p;
            n = T.so3() * n;

            // project point to reference to find correspondences
            Vec2 ip = ref.camera.project(p);

            // round to nearest integer
            ip = ip.array().round();

            int sx = ip(0);
            int sy = ip(1);



            double bestDist = std::numeric_limits<double>::infinity();
            Correspondence corr;

            // search in a small neighbourhood of the projection
            int S = params.searchRadius;
            for (int dy = -S; dy <= S; ++dy)
            {
                for (int dx = -S; dx <= S; ++dx)
                {
                    int x = sx + dx;
                    int y = sy + dy;

                    if (!ref.points.getConstImageView().inImage(y, x)) continue;

                    Vec3 p2 = ref.points(y, x);
                    Vec3 n2 = ref.normals(y, x);


                    if (!p2.allFinite() || !n2.allFinite()) continue;

                    auto distance = (p2 - p).norm();

                    auto depth    = p2(2);
                    auto invDepth = 1.0 / depth;

                    auto disTh = params.scaleDistanceThresByDepth ? params.distanceThres * depth : params.distanceThres;

                    if (distance < bestDist && distance < disTh && n.dot(n2) > params.cosNormalThres)
                    {
                        corr.refPoint  = p2;
                        corr.refNormal = n2;
                        corr.srcPoint  = p0;
                        corr.srcNormal = n0;
                        corr.weight    = params.useInvDepthAsWeight ? invDepth * invDepth : 1;
                        bestDist       = distance;
                    }
                }
            }

            if (std::isfinite(bestDist))
            {
                result.push_back(corr);
            }
        }
    }

    return result;
}

SE3 alignDepthMaps(DepthMap referenceDepthMap, DepthMap sourceDepthMap, const SE3& refPose, const SE3& srcPose,
                   const IntrinsicsPinholed& camera, int iterations, ProjectiveCorrespondencesParams params)
{
    DepthMapExtended ref(referenceDepthMap, camera, refPose);
    DepthMapExtended src(sourceDepthMap, camera, srcPose);


    AlignedVector<Correspondence> corrs;

    for (int k = 0; k < iterations; ++k)
    {
        corrs    = Saiga::ICP::projectiveCorrespondences(ref, src, params);
        src.pose = Saiga::ICP::pointToPlane(corrs, ref.pose, src.pose);
    }
    return src.pose;
}


}  // namespace ICP
}  // namespace Saiga
