/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once
#include "saiga/vision/ba/BABase.h"
#include "saiga/vision/scene/Scene.h"

#include "Recursive.h"

namespace Saiga
{
class SAIGA_VISION_API BARec : public BABase, public LMOptimizer
{
   public:
    // ============== Recusrive Matrix Types ==============
    static constexpr int blockSizeCamera = 6;
    static constexpr int blockSizePoint  = 3;
    using BlockBAScalar                  = double;

    using ADiag  = Eigen::Matrix<BlockBAScalar, blockSizeCamera, blockSizeCamera, Eigen::RowMajor>;
    using BDiag  = Eigen::Matrix<BlockBAScalar, blockSizePoint, blockSizePoint, Eigen::RowMajor>;
    using WElem  = Eigen::Matrix<BlockBAScalar, blockSizeCamera, blockSizePoint, Eigen::RowMajor>;
    using WTElem = Eigen::Matrix<BlockBAScalar, blockSizePoint, blockSizeCamera, Eigen::RowMajor>;
    using ARes   = Eigen::Matrix<BlockBAScalar, blockSizeCamera, 1>;
    using BRes   = Eigen::Matrix<BlockBAScalar, blockSizePoint, 1>;

    // Block structured diagonal matrices
    using UType = Eigen::DiagonalMatrix<Eigen::Recursive::MatrixScalar<ADiag>, -1>;
    using VType = Eigen::DiagonalMatrix<Eigen::Recursive::MatrixScalar<BDiag>, -1>;

    // Block structured vectors
    using DAType = Eigen::Matrix<Eigen::Recursive::MatrixScalar<ARes>, -1, 1>;
    using DBType = Eigen::Matrix<Eigen::Recursive::MatrixScalar<BRes>, -1, 1>;

    // Block structured sparse matrix
    using WType  = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<WElem>, Eigen::RowMajor>;
    using WTType = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<WTElem>, Eigen::RowMajor>;
    using SType  = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<ADiag>, Eigen::RowMajor>;


    using BAMatrix = Eigen::Recursive::SymmetricMixedMatrix2<UType, VType, WType>;
    using BAVector = Eigen::Recursive::MixedVector2<DAType, DBType>;
    using BASolver = Eigen::Recursive::MixedSymmetricRecursiveSolver<BAMatrix, BAVector>;

   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BARec() : BABase("Recursive BA") {}
    virtual ~BARec() {}
    virtual void create(Scene& scene) override { _scene = &scene; }

    // resserve space for n cameras and m points
    void reserve(int n, int m);

   private:
    Scene* _scene;

   private:
    int n, m;
    int totalN;     // with constant images
    int constantN;  // only constant images  n + constantN == totalN


    BAMatrix A;
    BAVector x, b, delta_x;
    BASolver solver;

    AlignedVector<SE3> x_u, oldx_u;
    AlignedVector<Vec3> x_v, oldx_v;

    // ============== Structure information ==============

    int observations;
    int schurEdges = 0;
    std::vector<std::vector<int>> schurStructure;

    // Number of seen world points for each camera + the corresponding exclusive scan and sum
    std::vector<int> cameraPointCounts, cameraPointCountsScan;
    // Number of observing cameras for each world point+ the corresponding exclusive scan and sum
    std::vector<int> pointCameraCounts, pointCameraCountsScan;


    struct ImageInfo
    {
        // id in the saiga.scene.image array
        int sceneImageId = -1;

        // compacted valid id. index into validImages array
        int validId = -1;

        // index into delta_x, A matrices.
        // this is -1 for constant cameras
        int variableId = -1;

        bool isConstant() { return variableId == -1; }
        bool isValid() { return validId >= 0; }
        explicit operator bool() { return isValid(); }
    };

    std::vector<ImageInfo> validImages;  // compact images + bool=constant
    std::vector<int> validPoints;
    std::vector<int> pointToValidMap;



    bool explizitSchur = false;
    bool computeWT     = true;

    Eigen::Recursive::LinearSolverOptions loptions;
    // ============= Multi Threading Stuff ===========
    //    int threads = 1;
    // each thread gets one vector
    std::vector<AlignedVector<BDiag>> pointDiagTemp;
    std::vector<AlignedVector<BRes>> pointResTemp;
    std::vector<double> localChi2;
    double chi2_sum;


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


}  // namespace Saiga
