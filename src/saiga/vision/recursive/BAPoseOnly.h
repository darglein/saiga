#pragma once


#include "saiga/vision/ba/BABase.h"
#include "saiga/vision/scene/Scene.h"

#include "Recursive.h"

namespace Saiga
{
class SAIGA_VISION_API BAPoseOnly : public BABase, public LMOptimizer
{
   public:
    static constexpr int blockSizePoint = 3;
    using T                             = double;

    using DiagType = Eigen::Matrix<T, blockSizePoint, blockSizePoint, Eigen::RowMajor>;
    using ResType  = Eigen::Matrix<T, blockSizePoint, 1>;

    /**
     * Optimize the camera extrinics of all cameras.
     * The world points are kept constant.
     *
     *
     */
    BAPoseOnly() : BABase("Point Only BA") {}
    virtual ~BAPoseOnly() {}

    virtual void create(Scene& scene) override { _scene = &scene; }

   private:
    int n;
    Scene* _scene;

    AlignedVector<DiagType> diagBlocks;
    AlignedVector<ResType> resBlocks;
    AlignedVector<Vec3> x_v, oldx_v;
    AlignedVector<Vec3> delta_x;

    // ============= Multi Threading Stuff ===========
    int threads = 1;
    std::vector<AlignedVector<DiagType>> diagTemp;
    std::vector<AlignedVector<ResType>> resTemp;
    std::vector<double> localChi2;
    // ============== LM Functions ==============

    virtual void init() override;
    virtual double computeQuadraticForm() override;
    virtual void addLambda(double lambda) override;
    virtual bool addDelta() override;
    virtual void revertDelta() override;
    virtual void solveLinearSystem() override;
    virtual double computeCost() override;
    virtual void finalize() override;

    virtual void setThreadCount(int n) override { threads = n; }
    virtual bool supportOMP() override { return true; }
};


}  // namespace Saiga
