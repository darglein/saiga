#include "PGORecursive.h"

#include "saiga/imgui/imgui.h"
#include "saiga/time/timer.h"
#include "saiga/util/Algorithm.h"
#include "saiga/vision/SparseHelper.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/kernels/PGO.h"
#include "saiga/vision/kernels/Robust.h"
#include "saiga/vision/recursiveMatrices/SparseCholesky.h"
#include "saiga/vision/recursiveMatrices/SparseInnerProduct.h"

#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"
#include "sophus/sim3.hpp"

#include <fstream>
#include <numeric>



namespace Saiga
{
void PGORec::initStructure(PoseGraph& scene)
{
    n = scene.poses.size();
    S.resize(n, n);
    b.resize(n);
    // assume no double edges + self edges
    //    S.reserve(scene.edges.size() + n);
}

void PGORec::compute(PoseGraph& scene)
{
    using T          = BlockBAScalar;
    using KernelType = Saiga::Kernel::PGO<T>;

    S.setZero();
    b.setZero();
    for (auto& e : scene.edges)
    {
        int i = e.from;
        int j = e.to;

        auto& target_ij = S(i, j).get();
        auto& target_ii = S(i, i).get();
        auto& target_jj = S(j, j).get();
        auto& target_ji = S(j, i).get();
        auto& target_ir = b(i).get();
        auto& target_jr = b(j).get();

        {
            KernelType::PoseJacobiType Jrowi, Jrowj;
            KernelType::ResidualType res;
            KernelType::evaluateResidualAndJacobian(scene.poses[i], scene.poses[j], e.meassurement.inverse(), res,
                                                    Jrowi, Jrowj);

            target_ij = Jrowi.transpose() * Jrowj;
            target_ji = target_ij.transpose();
            target_ii += Jrowi.transpose() * Jrowi;
            target_jj += Jrowj.transpose() * Jrowj;
            target_ir = Jrowi.transpose() * res;
            target_jr = Jrowj.transpose() * res;
        }
    }
    cout << expand(S) << endl << endl;
}


void PGORec::solve(PoseGraph& scene, const PGOOptions& options)
{
    initStructure(scene);
    compute(scene);
}



}  // namespace Saiga
