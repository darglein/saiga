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
class SAIGA_GLOBAL PGORec : public PGOBase
{
   public:
    PGORec() : PGOBase("recursive PGO") {}
    virtual ~PGORec() {}
    virtual void solve(PoseGraph& scene, const PGOOptions& options) override;

   private:
    int n;
    PSType S;
    PSDiagType Sdiag;
    PBType b;
    PBType x;

    double chi2;

    std::vector<std::pair<int, int>> edgeOffsets;
    PGOOptions options;
    void initStructure(PoseGraph& scene);
    void compute(PoseGraph& scene);
    void solveL(PoseGraph& scene);
};

}  // namespace Saiga
