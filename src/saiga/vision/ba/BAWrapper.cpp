/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "BAWrapper.h"

#ifdef SAIGA_USE_EIGENRECURSIVE
#    include "saiga/vision/recursive/BARecursive.h"
#endif

#ifdef SAIGA_USE_CERES
#    include "saiga/vision/ceres/CeresBA.h"
#endif

#ifdef SAIGA_USE_G2O
#    include "saiga/vision/g2o/g2oBA2.h"
#endif

#include "saiga/vision/scene/Scene.h"

namespace Saiga
{
BAWrapper::BAWrapper(const Framework& _fw) : fw(_fw)
{
    //    ba = std::unique_ptr<BABase>(new BARec);

    if (fw == Framework::Best)
    {
        fw = Framework::Recursive;
    }

    if (fw == Framework::Recursive)
    {
#ifdef SAIGA_USE_EIGENRECURSIVE
        ba = std::unique_ptr<BABase>(new BARec);
#endif
    }
    else if (fw == Framework::Ceres)
    {
#ifdef SAIGA_USE_CERES
        ba = std::unique_ptr<CeresBA>(new CeresBA);
#endif
    }
    else if (fw == Framework::G2O)
    {
#ifdef SAIGA_USE_G2O
        ba = std::unique_ptr<g2oBA2>(new g2oBA2);
#endif
    }


    SAIGA_ASSERT(ba);
}

void BAWrapper::create(Scene& scene)
{
    ba->create(scene);
}

OptimizationResults BAWrapper::solve(const OptimizationOptions& optimizationOptions, const BAOptions& baOptions)
{
    ba->baOptions              = baOptions;
    opt()->optimizationOptions = optimizationOptions;
    return opt()->solve();
}

OptimizationResults BAWrapper::initAndSolve(const OptimizationOptions& optimizationOptions, const BAOptions& baOptions)
{
    ba->baOptions              = baOptions;
    opt()->optimizationOptions = optimizationOptions;
    return opt()->initAndSolve();
}

Optimizer* BAWrapper::opt()
{
    if (fw == Framework::Recursive) return static_cast<BARec*>(ba.get());
#ifdef SAIGA_USE_CERES
    else if (fw == Framework::Ceres)
        return static_cast<CeresBA*>(ba.get());
#endif
#ifdef SAIGA_USE_G2O
    else if (fw == Framework::G2O)
        return static_cast<g2oBA2*>(ba.get());
#endif
    SAIGA_EXIT_ERROR("unknown framework.");
    return nullptr;
}



}  // namespace Saiga
