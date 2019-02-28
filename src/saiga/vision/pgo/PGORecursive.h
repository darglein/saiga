/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vision/pgo/PGOBase.h"
#include "saiga/vision/recursiveMatrices/RecursivePGOTemplates.h"

namespace Saiga
{
class SAIGA_VISION_API PGORec : public PGOBase, public LMOptimizer
{
   public:
    PGORec() : PGOBase("recursive PGO") {}
    virtual ~PGORec() {}
    //    virtual OptimizationResults solve() override;
    virtual void create(PoseGraph& scene) override { _scene = &scene; }


   private:
    int n;
    PSType S;
    PBType b;
    PBType delta_x;
    MixedSymmetricRecursiveSolver<PSType, PBType> solver;

    AlignedVector<SE3> x_u, oldx_u;

    double chi2;

    std::vector<int> edgeOffsets;
    PoseGraph* _scene;


    virtual void init() override;
    virtual double computeQuadraticForm() override;
    virtual void addLambda(double lambda) override;
    virtual void revertDelta() override;
    virtual void addDelta() override;
    virtual void solveLinearSystem() override;
    virtual double computeCost() override;
    virtual void finalize() override;
};

}  // namespace Saiga
