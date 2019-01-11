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

   private:
    int n, m;
    int observations;
    int schurEdges;
    UType U;
    VType V;
    WType W;
    WTType WT;

    DAType da;
    DBType db;

    DAType ea;
    DBType eb;

    // Solver tmps
    VType Vinv;
    WType Y;
    SType S;
    DAType ej;


    std::vector<std::vector<int>> schurStructure;

    void initStructure(Scene& scene);
    void computeUVW(Scene& scene);
};


}  // namespace Saiga
