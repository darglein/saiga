#include "PGORecursive.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/Algorithm.h"
#include "saiga/vision/HistogramImage.h"
#include "saiga/vision/LM.h"
#include "saiga/vision/kernels/PGO.h"
#include "saiga/vision/kernels/Robust.h"
#include "saiga/vision/recursiveMatrices/SparseSelfAdjoint.h"

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
    if (optimizationOptions.debugOutput)
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

    SAIGA_OPTIONAL_BLOCK_TIMER(optimizationOptions.debugOutput);
    using T          = BlockPGOScalar;
    using KernelType = Saiga::Kernel::PGO<T>;

    b.setZero();

    // set diagonal elements of S to zero
    for (int i = 0; i < n; ++i)
    {
        S.valuePtr()[S.outerIndexPtr()[i]].get().setZero();
    }

    double chi2 = 0;
    for (size_t k = 0; k < scene.edges.size(); ++k)
    {
        auto& e       = scene.edges[k];
        auto& offsets = edgeOffsets[k];
        int i         = e.from;
        int j         = e.to;

        auto& target_ij = S.valuePtr()[offsets].get();
        auto& target_ii = S.valuePtr()[S.outerIndexPtr()[i]].get();
        auto& target_jj = S.valuePtr()[S.outerIndexPtr()[j]].get();
        auto& target_ir = b(i).get();
        auto& target_jr = b(j).get();

        {
            KernelType::PoseJacobiType Jrowi, Jrowj;
            KernelType::ResidualType res;
            KernelType::evaluateResidualAndJacobian(x_u[i], x_u[j], e.meassurement.inverse(), res, Jrowi, Jrowj,
                                                    e.weight);

            if (scene.poses[i].constant) Jrowi.setZero();
            if (scene.poses[j].constant) Jrowj.setZero();

            // JtJ
            target_ij = Jrowi.transpose() * Jrowj;
            target_ii += Jrowi.transpose() * Jrowi;
            target_jj += Jrowj.transpose() * Jrowj;

            // Jtb
            target_ir -= Jrowi.transpose() * res;
            target_jr -= Jrowj.transpose() * res;


            auto c = res.squaredNorm();
            chi2 += c;
        }
    }


    if (optimizationOptions.debugOutput) cout << "chi2 " << chi2 << endl;
    return chi2;
}


double PGORec::computeCost()
{
    auto& scene = *_scene;

    SAIGA_OPTIONAL_BLOCK_TIMER(optimizationOptions.debugOutput);
    using T          = BlockPGOScalar;
    using KernelType = Saiga::Kernel::PGO<T>;



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

void PGORec::addDelta()
{
    oldx_u = x_u;

    for (int i = 0; i < n; ++i)
    {
        auto t = delta_x(i).get();
        x_u[i] = SE3::exp(t) * x_u[i];
    }
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

    int i = 0;
    for (auto& e : scene.poses)
    {
        if (!e.constant) e.se3 = x_u[i++];
    }
}


#if 0
OptimizationResults PGORec::solve()
{
    auto& scene = *_scene;


    init();

    LinearSolverOptions solverOptions;
    solverOptions.maxIterativeIterations = optimizationOptions.maxIterativeIterations;
    solverOptions.solverType             = LinearSolverOptions ::SolverType::Direct;

    MixedSymmetricRecursiveSolver<PSType, PBType> solver;


    for (int i = 0; i < optimizationOptions.maxIterations; ++i)
    {
        chi2 = computeQuadraticForm();
        addLambda(1e-4);

        solver.solve(S, delta_x, b, solverOptions);

        addDelta();

        // update scene
        //        for (size_t i = 0; i < scene.poses.size(); ++i)
        //        {
        //            Sophus::SE3<BlockPGOScalar>::Tangent t;
        //            t         = delta_x(i).get();
        //            auto& se3 = scene.poses[i].se3;
        //            se3       = Sophus::SE3d::exp(t.cast<double>()) * se3;
        //        }
    }

    finalize();

    OptimizationResults result;
    result.cost_final = chi2;
    return result;
}

#endif

}  // namespace Saiga
