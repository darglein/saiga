/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/vision/Optimizer.h"
#include "saiga/vision/recursiveMatrices/RecursiveMatrices.h"

#include "ArapProblem.h"
namespace Saiga
{
class RecursiveArap : public LMOptimizer
{
   public:
    ArapProblem* arap;

   protected:
    virtual void init() override;
    virtual double computeQuadraticForm() override;
    virtual void addLambda(double lambda) override;
    virtual void revertDelta() override;
    virtual void addDelta() override;
    virtual void solveLinearSystem() override;
    virtual double computeCost() override;
    virtual void finalize() override;

   private:
    static constexpr int BlockSize = 6;


    using T          = double;
    using PGOBlock   = Eigen::Matrix<T, BlockSize, BlockSize>;
    using PGOVector  = Eigen::Matrix<T, BlockSize, 1>;
    using PSType     = Eigen::SparseMatrix<MatrixScalar<PGOBlock>, Eigen::RowMajor>;
    using PSDiagType = Eigen::DiagonalMatrix<MatrixScalar<PGOBlock>, -1>;
    using PBType     = Eigen::Matrix<MatrixScalar<PGOVector>, -1, 1>;

    int n;
    PSType S;
    PBType b;
    PBType delta_x;
    MixedSymmetricRecursiveSolver<PSType, PBType> solver;
    AlignedVector<SE3> x_u, oldx_u;
    std::vector<int> edgeOffsets;
};

}  // namespace Saiga
