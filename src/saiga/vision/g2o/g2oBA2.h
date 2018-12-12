#pragma once

#include "saiga/vision/Scene.h"

namespace Saiga
{
class SAIGA_GLOBAL g2oBA2
{
   public:
    void optimize(Scene& scene, int its, double huberMono = -1, double huberStereo = -1);
};

}  // namespace Saiga
