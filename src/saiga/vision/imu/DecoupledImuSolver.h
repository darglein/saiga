/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/vision/recursive/Recursive.h"
#include "saiga/vision/util/Optimizer.h"

#include "DecoupledImuScene.h"
namespace Saiga::Imu
{
class SAIGA_VISION_API DecoupledImuSolver : public LMOptimizer
{
   public:
    // === API ===

    void Create(DecoupledImuScene& scene, const DecoupledImuScene::SolverOptions& params)
    {
        _scene       = &scene;
        this->params = params;
    }

   private:
    // ======== Constants ========
    static const int params_per_state = 9;
    // Gravity + Scale
    static const int global_params = 9;


    std::vector<int> states_without_preint;
    int N;
    int num_params;
    int non_zeros;
    std::vector<int> edgeOffsets;
    DecoupledImuScene* _scene;
    DecoupledImuScene::SolverOptions params;

    using BlockPGOScalar = double;
    using PGOBlock       = Eigen::Matrix<BlockPGOScalar, params_per_state, params_per_state>;
    using PGOVector      = Eigen::Matrix<BlockPGOScalar, params_per_state, 1>;
    using PSType         = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<PGOBlock>, Eigen::RowMajor>;
    using PSDiagType     = Eigen::DiagonalMatrix<Eigen::Recursive::MatrixScalar<PGOBlock>, -1>;
    using PBType         = Eigen::Matrix<Eigen::Recursive::MatrixScalar<PGOVector>, -1, 1>;
    using PGOSolver      = Eigen::Recursive::MixedSymmetricRecursiveSolver<PSType, PBType>;


    PSType S;
    PBType b;
    PGOSolver solver;

    //    Eigen::Matrix<double, -1, 1> x;
    PBType x;


    //    Eigen::Matrix<double, -1, -1> JtJ;
    //    Eigen::Matrix<double, -1, 1> Jtb;

    void RecomputePreint(bool always);

    // ============== LM Functions ==============

    virtual void init() override;
    virtual double computeQuadraticForm() override;
    virtual void addLambda(double lambda) override;
    virtual bool addDelta() override;
    virtual void revertDelta() override;
    virtual void solveLinearSystem() override;
    virtual double computeCost() override;
    virtual void finalize() override;
};

}  // namespace Saiga::Imu
