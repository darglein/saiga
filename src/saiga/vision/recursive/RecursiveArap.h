/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/vision/arap/ArapBase.h"
#include "saiga/vision/arap/ArapProblem.h"
#include "saiga/vision/util/Optimizer.h"

#include "Recursive.h"
namespace Saiga
{
class SAIGA_VISION_API RecursiveArap : public ArapBase, public LMOptimizer
{
   public:
    static constexpr int BlockSize = 6;


    using T          = double;
    using PGOBlock   = Eigen::Matrix<T, BlockSize, BlockSize>;
    using PGOVector  = Eigen::Matrix<T, BlockSize, 1>;
    using PSType     = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<PGOBlock>, Eigen::RowMajor>;
    using PSDiagType = Eigen::DiagonalMatrix<Eigen::Recursive::MatrixScalar<PGOBlock>, -1>;
    using PBType     = Eigen::Matrix<Eigen::Recursive::MatrixScalar<PGOVector>, -1, 1>;

   public:
    ArapProblem* arap;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    RecursiveArap() : ArapBase("Recursive") {}
    virtual void create(ArapProblem& scene) override { arap = &scene; }

   protected:
    virtual void init() override;
    virtual double computeQuadraticForm() override;
    virtual void addLambda(double lambda) override;
    virtual void revertDelta() override;
    virtual bool addDelta() override;
    virtual void solveLinearSystem() override;
    virtual double computeCost() override;
    virtual void finalize() override;

   private:
    int n;
    PSType S;
    PBType b;
    PBType delta_x;
    Eigen::Recursive::MixedSymmetricRecursiveSolver<PSType, PBType> solver;
    AlignedVector<SE3> x_u, oldx_u;
    std::vector<int> edgeOffsets;
};

}  // namespace Saiga
