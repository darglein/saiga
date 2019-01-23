#include "PGORecursive.h"

#include "saiga/imgui/imgui.h"
#include "saiga/time/timer.h"
#include "saiga/util/Algorithm.h"
#include "saiga/vision/HistogramImage.h"
#include "saiga/vision/SparseHelper.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/kernels/PGO.h"
#include "saiga/vision/kernels/Robust.h"
#include "saiga/vision/recursiveMatrices/LM.h"
#include "saiga/vision/recursiveMatrices/SparseCholesky.h"
#include "saiga/vision/recursiveMatrices/SparseInnerProduct.h"

#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"
#include "sophus/sim3.hpp"

#include <fstream>
#include <numeric>



#define NO_CG_SPEZIALIZATIONS
#define NO_CG_TYPES
using Scalar = Saiga::BlockBAScalar;
const int bn = Saiga::pgoBlockSizeCamera;
const int bm = Saiga::pgoBlockSizeCamera;
using Block  = Eigen::Matrix<Scalar, bn, bm>;
using Vector = Eigen::Matrix<Scalar, bn, 1>;

#include "saiga/vision/recursiveMatrices/CG.h"


namespace Saiga
{
void PGORec::initStructure(PoseGraph& scene)
{
    n = scene.poses.size();
    b.resize(n);
    Sdiag.resize(n);
    x.resize(n);

    // Compute structure of S
    S.resize(n, n);
    S.setZero();
    int N = scene.edges.size() * 2;
    S.reserve(N);

    // Compute outer structure pointer
    for (auto& e : scene.edges)
    {
        int i = e.from;
        int j = e.to;
        S.outerIndexPtr()[i]++;
        S.outerIndexPtr()[j]++;
    }

    exclusive_scan(S.outerIndexPtr(), S.outerIndexPtr() + S.outerSize(), S.outerIndexPtr(), 0);
    S.outerIndexPtr()[S.outerSize()] = N;


    // Precompute the offset in the sparse matrix for every edge
    edgeOffsets.reserve(scene.edges.size());
    std::vector<int> localOffsets(n, 0);
    for (auto& e : scene.edges)
    {
        int i  = e.from;
        int j  = e.to;
        int li = localOffsets[i]++;
        int lj = localOffsets[j]++;

        int offseti = S.outerIndexPtr()[i] + li;
        int offsetj = S.outerIndexPtr()[j] + lj;

        S.innerIndexPtr()[offseti] = j;
        S.innerIndexPtr()[offsetj] = i;


        edgeOffsets.emplace_back(offseti, offsetj);
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
    using T          = BlockBAScalar;
    using KernelType = Saiga::Kernel::PGO<T>;

    b.setZero();
    Sdiag.setZero();

    chi2 = 0;
    for (int k = 0; k < scene.edges.size(); ++k)
    {
        auto& e       = scene.edges[k];
        auto& offsets = edgeOffsets[k];
        int i         = e.from;
        int j         = e.to;

        PGOBlock target_ij;
        PGOBlock target_ji;
        auto& target_ii = Sdiag.diagonal()(i).get();
        auto& target_jj = Sdiag.diagonal()(j).get();

        auto& target_ir = b(i).get();
        auto& target_jr = b(j).get();

        {
            KernelType::PoseJacobiType Jrowi, Jrowj;
            KernelType::ResidualType res;
            KernelType::evaluateResidualAndJacobian(scene.poses[i].se3, scene.poses[j].se3, e.meassurement.inverse(),
                                                    res, Jrowi, Jrowj);

            target_ij = Jrowi.transpose() * Jrowj;
            target_ji = target_ij.transpose();
            target_ii += Jrowi.transpose() * Jrowi;
            target_jj += Jrowj.transpose() * Jrowj;
            target_ir += Jrowi.transpose() * res;
            target_jr += Jrowj.transpose() * res;

            S.valuePtr()[offsets.first]  = target_ij;
            S.valuePtr()[offsets.second] = target_ji;

            auto c = res.squaredNorm();
            chi2 += c;
        }
    }

    applyLMDiagonal(Sdiag);
    //    applyLMDiagonalG2O(Sdiag, 1);

    if (options.debugOutput) cout << "chi2 " << chi2 << endl;
}

void PGORec::solveL(PoseGraph& scene)
{
    SAIGA_OPTIONAL_BLOCK_TIMER(options.debugOutput);
    if (options.solverType == PGOOptions::SolverType::Direct)

    {
        x.setZero();

        Eigen::MatrixXd S3 = expand(S) + expand(Sdiag);
        Eigen::VectorXd x2 = S3.ldlt().solve(expand(b));

        // copy back into da
        for (int i = 0; i < n; ++i)
        {
            x(i) = x2.segment<pgoBlockSizeCamera>(i * pgoBlockSizeCamera);
        }

        //        cout << expand(x).transpose() << endl;
        //        cout << "Direct Residual: " << (expand(S) * expand(x) - expand(b)).norm() << endl;
    }
    else
    {
        x.setZero();
        RecursiveDiagonalPreconditioner<MatrixScalar<PGOBlock>> P;
        Eigen::Index iters = options.maxIterativeIterations;
        Scalar tol         = options.iterativeTolerance;


        P.compute(Sdiag);
        PBType tmp(n);
        recursive_conjugate_gradient(
            [&](const PBType& v) {
                //                tmp = S * v;
                tmp = Sdiag.diagonal().array() * v.array();
                tmp = S * v + tmp;
                return tmp;
            },
            b, x, P, iters, tol);

        if (options.debugOutput) cout << S.nonZeros() << " error " << tol << " iterations " << iters << endl;
        //        cout << expand(x).transpose() << endl;
        //        cout << "Iterative Residual: " << (expand(S) * expand(x) - expand(b)).norm() << endl;
    }

    //    cout << expand(x).transpose() << endl;

    for (size_t i = 0; i < scene.poses.size(); ++i)
    {
        Sophus::SE3<BlockBAScalar>::Tangent t;
        t         = x(i).get();
        auto& se3 = scene.poses[i].se3;
        se3       = Sophus::SE3d::exp(t.cast<double>()) * se3;
    }
}


void PGORec::solve(PoseGraph& scene, const PGOOptions& options)
{
    this->options = options;
    initStructure(scene);
    for (int i = 0; i < options.maxIterations; ++i)
    {
        compute(scene);
        solveL(scene);
    }
}



}  // namespace Saiga
