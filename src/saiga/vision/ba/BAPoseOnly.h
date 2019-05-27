#pragma once


#include "BABase.h"

namespace Saiga
{
class SAIGA_VISION_API BAPoseOnly : public BABase, public Optimizer
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
    void posePointSparseSchur(Scene& scene);


    virtual OptimizationResults initAndSolve() override;
    virtual void create(Scene& scene) override { _scene = &scene; }

   private:
    Scene* _scene;
};


}  // namespace Saiga
