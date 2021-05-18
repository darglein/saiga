/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vision/pgo/PGOBase.h"

#include "Recursive.h"

namespace Saiga
{
class SAIGA_VISION_API PGORec : public PGOBase, public LMOptimizer
{
   public:
    using PGOTransformation = SE3;
    // ============== Recusrive Matrix Types ==============

    static constexpr int pgoBlockSizeCamera = PGOTransformation::DoF;
    using BlockPGOScalar                    = double;

    using PGOBlock   = Eigen::Matrix<BlockPGOScalar, pgoBlockSizeCamera, pgoBlockSizeCamera>;
    using PGOVector  = Eigen::Matrix<BlockPGOScalar, pgoBlockSizeCamera, 1>;
    using PSType     = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<PGOBlock>, Eigen::RowMajor>;
    using PSDiagType = Eigen::DiagonalMatrix<Eigen::Recursive::MatrixScalar<PGOBlock>, -1>;
    using PBType     = Eigen::Matrix<Eigen::Recursive::MatrixScalar<PGOVector>, -1, 1>;

    using PGOSolver = Eigen::Recursive::MixedSymmetricRecursiveSolver<PSType, PBType>;

   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PGORec() : PGOBase("recursive PGO") {}
    virtual ~PGORec() {}
    virtual void create(PoseGraph& scene) override { _scene = &scene; }


   private:
    int n;
    PSType S;
    PBType b;
    PBType delta_x;
    PGOSolver solver;



    AlignedVector<PGOTransformation> x_u, oldx_u;


    std::vector<int> edgeOffsets;
    PoseGraph* _scene;

    // ============== LM Functions ==============

    virtual void init() override;
    virtual double computeQuadraticForm() override;
    virtual void addLambda(double lambda) override;
    virtual void revertDelta() override;
    virtual bool addDelta() override;
    virtual void solveLinearSystem() override;
    virtual double computeCost() override;
    virtual void finalize() override;
};

}  // namespace Saiga
