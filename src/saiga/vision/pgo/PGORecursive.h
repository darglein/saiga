/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vision/pgo/PGOBase.h"

#include "RecursivePGOTemplates.h"

namespace Saiga
{
class SAIGA_GLOBAL PGORec : public PGOBase
{
   public:
    PGORec() : PGOBase("recursive PGO") {}
    virtual void solve(PoseGraph& scene, const PGOOptions& options) override;

   private:
    int n;
    PSType S;
    PBType b;

    void initStructure(PoseGraph& scene);
    void compute(PoseGraph& scene);
};

}  // namespace Saiga
