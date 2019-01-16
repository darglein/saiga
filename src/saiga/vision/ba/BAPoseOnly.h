#pragma once


#include "BABase.h"

namespace Saiga
{
class SAIGA_GLOBAL BAPoseOnly : public BABase
{
   public:
    /**
     * Optimize the camera extrinics of all cameras.
     * The world points are kept constant.
     *
     *
     */
    BAPoseOnly() : BABase("Simple Sparse BA") {}
    void poseOnlySparse(Scene& scene, int its);
    void posePointDense(Scene& scene, int its);
    void posePointSparse(Scene& scene, int its);


    virtual void solve(Scene& scene, const BAOptions& options) override;
};


}  // namespace Saiga
