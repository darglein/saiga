#pragma once

#include "saiga/vision/Scene.h"

namespace Saiga
{
class BAPoseOnly
{
   public:
    /**
     * Optimize the camera extrinics of all cameras.
     * The world points are kept constant.
     *
     *
     */
    void poseOnlySparse(Scene& scene, int its);
    void posePointDense(Scene& scene, int its);
    void posePointSparse(Scene& scene, int its);
    void posePointDenseBlock(Scene& scene, int its);
};


}  // namespace Saiga
