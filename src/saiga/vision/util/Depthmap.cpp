/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "Depthmap.h"


namespace Saiga
{
namespace Depthmap
{
void toPointCloud(DepthMap dm, DepthPointCloud pc, const IntrinsicsPinholed& camera)
{
    SAIGA_ASSERT(dm.h == pc.h && dm.w == pc.w);
    for (int i = 0; i < dm.h; ++i)
    {
        for (int j = 0; j < dm.w; ++j)
        {
            Vec2 ip(j, i);
            auto depth = dm(i, j);

            Vec3 result;
            if (depth > 0)
                result = camera.unproject(ip, depth);
            else
                result = infinityVec3();
            pc(i, j) = result;
        }
    }
}


void normalMap(DepthPointCloud pc, DepthNormalMap normals)
{
    SAIGA_ASSERT(normals.h == pc.h && normals.w == pc.w);
    for (int i = 0; i < normals.h; ++i)
    {
        for (int j = 0; j < normals.w; ++j)
        {
            // center, left, right, up, down
            auto c = pc(i, j);
            if (!c.allFinite())
            {
                normals(i, j) = infinityVec3();
                continue;
            }

            auto l = pc.clampedRead(i - 1, j);
            auto r = pc.clampedRead(i + 1, j);
            auto u = pc.clampedRead(i, j - 1);
            auto d = pc.clampedRead(i, j + 1);

            if (l.allFinite() && r.allFinite() && u.allFinite() && d.allFinite())
            {
                Vec3 rl       = r - l;
                Vec3 ud       = u - d;
                Vec3 result   = (ud.cross(rl)).normalized();
                normals(i, j) = result;
            }
            else
            {
                normals(i, j) = infinityVec3();
            }
        }
    }
}


}  // namespace Depthmap
}  // namespace Saiga
