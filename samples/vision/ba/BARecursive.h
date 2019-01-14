#pragma once

#include "saiga/vision/BlockRecursiveBATemplates.h"
#include "saiga/vision/Scene.h"
namespace Saiga
{
class BARec
{
   public:
    /**
     * Optimize the camera extrinics of all cameras.
     * The world points are kept constant.
     *
     *
     */
    void solve(Scene& scene, int its);

    void imgui();

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



    bool iterativeSolver = true;
    bool explizitSchur   = true;
    bool computeWT       = false;
    void initStructure(Scene& scene);
    void computeUVW(Scene& scene);
};


}  // namespace Saiga
