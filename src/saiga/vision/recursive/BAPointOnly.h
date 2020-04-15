#pragma once


#include "saiga/vision/ba/BABase.h"
#include "saiga/vision/scene/Scene.h"

#include "Recursive.h"

namespace Saiga
{
class SAIGA_VISION_API BAPointOnly : public BABase
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
    BAPointOnly() : BABase("Point Only BA") {}
    virtual ~BAPointOnly() {}

    virtual void create(Scene& scene) { _scene = &scene; }

    OptimizationOptions optimizationOptions;


    OptimizationResults initAndSolve();

   private:
    int n;
    Scene* _scene;

    AlignedVector<DiagType> diagBlocks;
    AlignedVector<ResType> resBlocks;
    AlignedVector<Vec3> x_v, oldx_v;
    AlignedVector<Vec3> delta_x;

    std::vector<double> chi2_per_point, chi2_per_point_new;
    // ============= Multi Threading Stuff ===========
    int threads = 1;
    std::vector<AlignedVector<DiagType>> diagTemp;
    std::vector<AlignedVector<ResType>> resTemp;
    // ============== LM Functions ==============

    virtual void init();
    OptimizationResults solve();
    virtual double computeQuadraticForm();
    virtual void addLambda(double lambda);
    virtual bool addDelta();
    virtual void solveLinearSystem();
    virtual double computeCost();
    virtual void finalize();

    virtual void setThreadCount(int n) { threads = n; }
    virtual bool supportOMP() { return true; }
};


}  // namespace Saiga
