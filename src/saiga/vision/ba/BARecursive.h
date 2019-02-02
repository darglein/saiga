/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "BABase.h"
//#define RECURSIVE_BA_VECTORIZE
//#define RECURSIVE_BA_FLOAT
#define RECURSIVE_BA_USE_TIMERS false
#include "saiga/vision/recursiveMatrices/BlockRecursiveBATemplates.h"
namespace Saiga
{
class SAIGA_VISION_API BARec : public BABase
{
   public:
    BARec() : BABase("Recursive BA") {}
    virtual ~BARec() {}
    virtual void solve(Scene& scene, const BAOptions& options) override;

   private:
    // ==== Structure information ====
    int n, m;
    int observations;
    int schurEdges;
    std::vector<std::vector<int>> schurStructure;

    // Number of seen world points for each camera + the corresponding exclusive scan and sum
    std::vector<int> cameraPointCounts, cameraPointCountsScan;
    // Number of observing cameras for each world point+ the corresponding exclusive scan and sum
    std::vector<int> pointCameraCounts, pointCameraCountsScan;

    // ==== Main (recursive) Variables ====
    UType U;
    VType V;
    WType W;
    WTType WT;

    DAType da;
    DBType db;
    DAType ea;
    DBType eb;

    // ==== Solver tmps ====
    DBType q;
    VType Vinv;
    WType Y;
    SType S;
    Eigen::DiagonalMatrix<MatrixScalar<ADiag>, -1> Sdiag;
    DAType ej;


    //    std::vector<int> imageIds;
    std::vector<int> validImages;
    std::vector<int> validPoints;
    std::vector<int> pointToValidMap;

    BAOptions options;

    double chi2;
    void initStructure(Scene& scene);
    void computeUVW(Scene& scene);
    void computeSchur();
    void solveSchur();
    void finalizeSchur();
    void updateScene(Scene& scene);

    bool explizitSchur = false;
    bool computeWT     = true;
};


}  // namespace Saiga
