#include "PGORecursive.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/Algorithm.h"
#include "saiga/vision/kernels/PGO.h"
#include "saiga/vision/kernels/Robust.h"
#include "saiga/vision/util/HistogramImage.h"
#include "saiga/vision/util/LM.h"

#include <fstream>
#include <numeric>



namespace Saiga
{
void PGORec::init()
{
    auto& scene = *_scene;

    n = scene.poses.size();
    b.resize(n);
    delta_x.resize(n);

    x_u.resize(n);
    oldx_u.resize(n);

    // Make a copy of the initial parameters
    int i = 0;
    for (auto& e : scene.poses)
    {
        x_u[i++] = e.se3;
    }

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


        edgeOffsets.emplace_back(offseti);
    }

    // Create a sparsity histogram
    if (false && optimizationOptions.debugOutput)
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

double PGORec::computeQuadraticForm()
{
    auto& scene = *_scene;

    //    SAIGA_BLOCK_TIMER();
    //    SAIGA_OPTIONAL_BLOCK_TIMER(optimizationOptions.debugOutput);
    // using T          = BlockPGOScalar;
    using KernelType = Saiga::Kernel::PGO<PGOTransformation>;

    b.setZero();

    // set diagonal elements of S to zero
    for (int i = 0; i < n; ++i)
    {
        S.valuePtr()[S.outerIndexPtr()[i]].get().setZero();
    }

    double chi2 = 0;
    //#pragma omp parallel

    //        auto b2 = b;
    AlignedVector<PGOBlock> diagBlocks(n);
    for (int i = 0; i < n; ++i)
    {
        diagBlocks[i].setZero();
    }
    double chi2local = 0;
    //#pragma omp for
    for (size_t k = 0; k < scene.edges.size(); ++k)
    {
        auto& e       = scene.edges[k];
        auto& offsets = edgeOffsets[k];
        int i         = e.from;
        int j         = e.to;

        auto& target_ij = S.valuePtr()[offsets].get();
        //            auto& target_ii = S.valuePtr()[S.outerIndexPtr()[i]].get();
        //            auto& target_jj = S.valuePtr()[S.outerIndexPtr()[j]].get();
        auto& target_ii = diagBlocks[i];
        auto& target_jj = diagBlocks[j];
        auto& target_ir = b(i).get();
        auto& target_jr = b(j).get();

        {
            KernelType::PoseJacobiType Jrowi, Jrowj;
            KernelType::ResidualType res;
            KernelType::evaluateResidualAndJacobian(x_u[i], x_u[j], e.meassurement.inverse(), res, Jrowi, Jrowj,
                                                    e.weight);

            if (scene.poses[i].constant) Jrowi.setZero();
            if (scene.poses[j].constant) Jrowj.setZero();

            auto c = res.squaredNorm();

            // JtJ
            target_ij = Jrowi.transpose() * Jrowj;

            auto ii = (Jrowi.transpose() * Jrowi).eval();
            auto jj = (Jrowj.transpose() * Jrowj).eval();
            //#pragma omp critical
            {
                target_ii += ii;
                target_jj += jj;
            }
            auto ir = Jrowi.transpose() * res;
            auto jr = Jrowj.transpose() * res;
            //#pragma omp critical
            {
                // Jtb
                target_ir -= ir;
                target_jr -= jr;
            }



            chi2local += c;
        }
    }

    //#pragma omp atomic
    chi2 += chi2local;
    //#pragma omp critical
    {
        //            b += b2;
    }

    for (int i = 0; i < n; ++i)
    {
        //#pragma omp critical
        S.valuePtr()[S.outerIndexPtr()[i]].get() += diagBlocks[i];
    }



    //    if (optimizationOptions.debugOutput) std::cout << "chi2 " << chi2 << std::endl;
    return chi2;
}


double PGORec::computeCost()
{
    auto& scene = *_scene;

    //    SAIGA_OPTIONAL_BLOCK_TIMER(optimizationOptions.debugOutput);
    // using T          = BlockPGOScalar;
    using KernelType = Saiga::Kernel::PGO<PGOTransformation>;



    double chi2 = 0;
    for (size_t k = 0; k < scene.edges.size(); ++k)
    {
        auto& e = scene.edges[k];
        int i   = e.from;
        int j   = e.to;


        {
            KernelType::PoseJacobiType Jrowi, Jrowj;
            KernelType::ResidualType res;
            KernelType::evaluateResidual(x_u[i], x_u[j], e.meassurement.inverse(), res, e.weight);

            auto c = res.squaredNorm();
            chi2 += c;
        }
    }
    return chi2;
}

void PGORec::addLambda(double lambda)
{
    // apply lm
    for (int i = 0; i < n; ++i)
    {
        auto& d = S.valuePtr()[S.outerIndexPtr()[i]].get();
        applyLMDiagonalInner(d, lambda);
    }
}

bool PGORec::addDelta()
{
    auto& scene = *_scene;
    oldx_u      = x_u;

    for (int i = 0; i < n; ++i)
    {
        if (scene.poses[i].constant) continue;
        auto t = delta_x(i).get();
#ifdef PGO_SIM3
        if (scene.fixScale) t[6] = 0;
#endif

        //        std::cout << t.transpose() << std::endl;
        x_u[i] = PGOTransformation::exp(t) * x_u[i];
    }
    return true;
}

void PGORec::solveLinearSystem()
{
    using namespace Eigen::Recursive;

    LinearSolverOptions loptions;

    loptions.maxIterativeIterations = optimizationOptions.maxIterativeIterations;
    loptions.iterativeTolerance     = optimizationOptions.iterativeTolerance;

    loptions.solverType = (optimizationOptions.solverType == OptimizationOptions::SolverType::Direct)
                              ? LinearSolverOptions::SolverType::Direct
                              : LinearSolverOptions::SolverType::Iterative;


    solver.solve(S, delta_x, b, loptions);
}

void PGORec::revertDelta()
{
    x_u = oldx_u;
}
void PGORec::finalize()
{
    auto& scene = *_scene;

    //    int i = 0;
    //    for (auto& e : scene.poses)
    //    {
    for (int i = 0; i < n; ++i)
    {
        auto& p = scene.poses[i];
        if (!p.constant) p.se3 = x_u[i];
    }
}

}  // namespace Saiga
