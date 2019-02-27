#include "PGORecursive.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/Algorithm.h"
#include "saiga/vision/HistogramImage.h"
#include "saiga/vision/kernels/PGO.h"
#include "saiga/vision/kernels/Robust.h"
#include "saiga/vision/recursiveMatrices/SparseSelfAdjoint.h"

#include <fstream>
#include <numeric>



namespace Saiga
{
void PGORec::initStructure(PoseGraph& scene)
{
    n = scene.poses.size();
    b.resize(n);
    x.resize(n);

    // Compute structure of S
    S.resize(n, n);
    S.setZero();
    int N = scene.edges.size() + n;
    S.reserve(N);

    // Compute outer structure pointer
    for (auto& e : scene.edges)
    {
        int i = e.from;
        int j = e.to;
        SAIGA_ASSERT(i != j);
        SAIGA_ASSERT(i < j);

        S.outerIndexPtr()[i]++;
    }

    // make room for the diagonal element
    for (int i = 0; i < n; ++i)
    {
        S.outerIndexPtr()[i]++;
    }

    exclusive_scan(S.outerIndexPtr(), S.outerIndexPtr() + S.outerSize(), S.outerIndexPtr(), 0);
    S.outerIndexPtr()[S.outerSize()] = N;


    // insert diagonal index
    for (int i = 0; i < n; ++i)
    {
        int offseti                = S.outerIndexPtr()[i];
        S.innerIndexPtr()[offseti] = i;
    }

    // Precompute the offset in the sparse matrix for every edge
    edgeOffsets.reserve(scene.edges.size());
    std::vector<int> localOffsets(n, 1);
    for (auto& e : scene.edges)
    {
        int i = e.from;
        int j = e.to;

        int li = localOffsets[i]++;

        int offseti = S.outerIndexPtr()[i] + li;

        S.innerIndexPtr()[offseti] = j;


        edgeOffsets.emplace_back(offseti, 0);
    }

    // Create a sparsity histogram
    if (options.debugOutput)
    {
        HistogramImage img(n, n, 1024, 1024);
        for (auto& e : scene.edges)
        {
            int i = e.from;
            int j = e.to;
            img.add(i, j, 1);
            img.add(j, i, 1);
            img.add(i, i, 1);
            img.add(j, j, 1);
        }
        img.writeBinary("vision_pgo_sparsity.png");
    }
}

void PGORec::compute(PoseGraph& scene)
{
    SAIGA_OPTIONAL_BLOCK_TIMER(options.debugOutput);
    using T          = BlockPGOScalar;
    using KernelType = Saiga::Kernel::PGO<T>;

    b.setZero();

    // set diagonal elements of S to zero
    for (int i = 0; i < n; ++i)
    {
        S.valuePtr()[S.outerIndexPtr()[i]].get().setZero();
    }

    chi2 = 0;
    for (size_t k = 0; k < scene.edges.size(); ++k)
    {
        auto& e       = scene.edges[k];
        auto& offsets = edgeOffsets[k];
        int i         = e.from;
        int j         = e.to;

        PGOBlock target_ij;
        PGOBlock target_ji;

        auto& target_ir = b(i).get();
        auto& target_jr = b(j).get();

        {
            KernelType::PoseJacobiType Jrowi, Jrowj;
            KernelType::ResidualType res;
            KernelType::evaluateResidualAndJacobian(scene.poses[i].se3, scene.poses[j].se3, e.meassurement.inverse(),
                                                    res, Jrowi, Jrowj);

            if (scene.poses[i].constant) Jrowi.setZero();
            if (scene.poses[j].constant) Jrowj.setZero();

            target_ij = Jrowi.transpose() * Jrowj;
            target_ji = target_ij.transpose();
            target_ir -= Jrowi.transpose() * res;
            target_jr -= Jrowj.transpose() * res;

            S.valuePtr()[offsets.first] = target_ij;

            S.valuePtr()[S.outerIndexPtr()[i]].get() += Jrowi.transpose() * Jrowi;
            S.valuePtr()[S.outerIndexPtr()[j]].get() += Jrowj.transpose() * Jrowj;


            auto c = res.squaredNorm();
            chi2 += c;
        }
    }

    // apply lm
    for (int i = 0; i < n; ++i)
    {
        auto& d = S.valuePtr()[S.outerIndexPtr()[i]].get();
        applyLMDiagonalInner(d);
    }

    if (options.debugOutput) cout << "chi2 " << chi2 << endl;
}



void PGORec::solve(PoseGraph& scene, const PGOOptions& options)
{
    this->options = options;
    initStructure(scene);

    LinearSolverOptions solverOptions;
    solverOptions.maxIterativeIterations = options.maxIterativeIterations;
    solverOptions.solverType             = LinearSolverOptions ::SolverType::Direct;

    MixedSymmetricRecursiveSolver<PSType, PBType> solver;


    for (int i = 0; i < options.maxIterations; ++i)
    {
        compute(scene);

        solver.solve(S, x, b, solverOptions);

        // update scene
        for (size_t i = 0; i < scene.poses.size(); ++i)
        {
            Sophus::SE3<BlockPGOScalar>::Tangent t;
            t         = x(i).get();
            auto& se3 = scene.poses[i].se3;
            se3       = Sophus::SE3d::exp(t.cast<double>()) * se3;
        }
    }
}



}  // namespace Saiga
