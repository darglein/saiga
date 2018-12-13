#pragma once

#include "saiga/vision/Scene.h"

namespace Saiga
{
class BAPoseOnly
{
   public:
    void optimize(Scene& scene, int its);
};


}  // namespace Saiga
