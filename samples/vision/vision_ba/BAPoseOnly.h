#pragma once

#include "saiga/vision/Scene.h"

namespace Saiga
{
class BAPoseOnly
{
   public:
    void poseOnlyDense(Scene& scene, int its);
    void posePointDense(Scene& scene, int its);
};


}  // namespace Saiga
