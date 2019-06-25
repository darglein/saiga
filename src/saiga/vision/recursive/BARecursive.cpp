/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "BARecursive.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/Algorithm.h"
#include "saiga/vision/HistogramImage.h"
#include "saiga/vision/LM.h"
#include "saiga/vision/kernels/BAPose.h"
#include "saiga/vision/kernels/BAPosePoint.h"
#include "saiga/vision/kernels/Robust.h"

#include <fstream>
#include <numeric>

#define RECURSIVE_BA_USE_TIMERS false

namespace Saiga
{
void BARec::init()
{
    //    std::cout << "Test sizes: " << sizeof(Scene) << " " << sizeof(BARec)<< " " << sizeof(BABase)<< " " <<
    //    sizeof(LMOptimizer) << std::endl; std::cout << "Test sizes2: " << sizeof(BAMatrix) << " " <<
    //    sizeof(BAVector)<< " " << sizeof(BASolver)<< " " << sizeof(AlignedVector<SE3>) << std::endl;

    Scene& scene = *_scene;

    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && optimizationOptions.debugOutput);


    // currently the scene must be in a valid state
    SAIGA_ASSERT(scene);

    if (optimizationOptions.solverType == OptimizationOptions::SolverType::Direct)
    {
        explizitSchur = true;
        computeWT     = true;
    }
    else
    {
        explizitSchur = false;
        computeWT     = true;
    }



    // Check how many valid and cameras exist and construct the compact index sets
    validPoints.clear();
    validImages.clear();
    pointToValidMap.clear();
    validImages.reserve(scene.images.size());
    validPoints.reserve(scene.worldPoints.size());
    pointToValidMap.resize(scene.worldPoints.size());

    for (int i = 0; i < (int)scene.images.size(); ++i)
    {
        auto& img = scene.images[i];
        if (!img) continue;
        validImages.push_back(i);
    }

    for (int i = 0; i < (int)scene.worldPoints.size(); ++i)
    {
        auto& wp = scene.worldPoints[i];
        if (!wp) continue;
        int validId        = validPoints.size();
        pointToValidMap[i] = validId;
        validPoints.push_back(i);
    }

    n = validImages.size();
    m = validPoints.size();


    SAIGA_ASSERT(n > 0 && m > 0);

    A.resize(n, m);
    //    U.resize(n);
    //    V.resize(m);

    delta_x.resize(n, m);
    b.resize(n, m);

    x_u.resize(n);
    oldx_u.resize(n);
    x_v.resize(m);
    oldx_v.resize(m);


    // Make a copy of the initial parameters
    for (int i = 0; i < (int)validImages.size(); ++i)
    {
        auto& img = scene.images[validImages[i]];
        x_u[i]    = scene.extrinsics[img.extr].se3;
    }

    for (int i = 0; i < (int)validPoints.size(); ++i)
    {
        auto& wp = scene.worldPoints[validPoints[i]];
        x_v[i]   = wp.p;
    }

    cameraPointCounts.clear();
    cameraPointCounts.resize(n, 0);
    cameraPointCountsScan.resize(n);
    pointCameraCounts.clear();
    pointCameraCounts.resize(m, 0);
    pointCameraCountsScan.resize(m);
    observations = 0;
    for (int i = 0; i < (int)validImages.size(); ++i)
    {
        auto& img = scene.images[validImages[i]];
        for (auto& ip : img.stereoPoints)
        {
            if (ip.wp == -1) continue;

            int j = pointToValidMap[ip.wp];
            cameraPointCounts[i]++;
            pointCameraCounts[j]++;
            observations++;
        }
    }

    auto test1 =
        Saiga::exclusive_scan(cameraPointCounts.begin(), cameraPointCounts.end(), cameraPointCountsScan.begin(), 0);
    auto test2 =
        Saiga::exclusive_scan(pointCameraCounts.begin(), pointCameraCounts.end(), pointCameraCountsScan.begin(), 0);

    SAIGA_ASSERT(test1 == observations && test2 == observations);

    // preset the outer matrix structure
    //    W.resize(n, m);
    A.w.setZero();
    A.w.reserve(observations);

    for (int k = 0; k < A.w.outerSize(); ++k)
    {
        A.w.outerIndexPtr()[k] = cameraPointCountsScan[k];
    }
    A.w.outerIndexPtr()[A.w.outerSize()] = observations;



    // Create sparsity histogram of the schur complement
    if (false && optimizationOptions.debugOutput)
    {
        HistogramImage img(n, n, 512, 512);
        schurStructure.clear();
        schurStructure.resize(n, std::vector<int>(n, -1));
        for (auto& wp : scene.worldPoints)
        {
            for (auto& ref : wp.stereoreferences)
            {
                for (auto& ref2 : wp.stereoreferences)
                {
                    int i1 = validImages[ref.first];
                    int i2 = validImages[ref2.first];

                    schurStructure[i1][ref2.first] = ref2.first;
                    schurStructure[i2][ref.first]  = ref.first;
                    img.add(i1, ref2.first, 1);
                    img.add(i2, ref.first, 1);
                }
            }
        }

        // compact it
        schurEdges = 0;
        for (auto& v : schurStructure)
        {
            v.erase(std::remove(v.begin(), v.end(), -1), v.end());
            schurEdges += v.size();
        }
        img.writeBinary("vision_ba_schur_sparsity.png");
    }


    // Create a sparsity histogram of the complete matrix
    if (false && optimizationOptions.debugOutput)
    {
        HistogramImage img(n + m, n + m, 512, 512);
        for (int i = 0; i < n + m; ++i)
        {
            img.add(i, i, 1);
        }

        for (int i = 0; i < (int)validImages.size(); ++i)
        {
            auto& img2 = scene.images[validImages[i]];
            for (auto& ip : img2.stereoPoints)
            {
                if (!ip) continue;

                int j   = pointToValidMap[ip.wp];
                int iid = validImages[i];
                img.add(iid, j + n, 1);
                img.add(j + n, iid, 1);
            }
        }
        img.writeBinary("vision_ba_complete_sparsity.png");
    }


    if (optimizationOptions.debugOutput)
    {
        std::cout << "." << std::endl;
        std::cout << "Structure Analyzed." << std::endl;
        std::cout << "Cameras: " << n << std::endl;
        std::cout << "Points: " << m << std::endl;
        std::cout << "Observations: " << observations << std::endl;
#if 1
        std::cout << "Schur Edges: " << schurEdges << std::endl;
        std::cout << "Non Zeros LSE: " << schurEdges * 6 * 6 << std::endl;
        std::cout << "Density: " << double(schurEdges * 6.0 * 6) / double(double(n) * n * 6 * 6) * 100 << "%"
                  << std::endl;
#endif

#if 1
        // extended analysis
        double averageCams   = 0;
        double averagePoints = 0;
        for (auto i : cameraPointCounts) averagePoints += i;
        averagePoints /= cameraPointCounts.size();
        for (auto i : pointCameraCounts) averageCams += i;
        averageCams /= pointCameraCounts.size();
        std::cout << "Average Points per Camera: " << averagePoints << std::endl;
        std::cout << "Average Cameras per Point: " << averageCams << std::endl;
#endif
        std::cout << "." << std::endl;
    }
}

double BARec::computeQuadraticForm()
{
    Scene& scene = *_scene;

    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && optimizationOptions.debugOutput);

    using T          = BlockBAScalar;
    using KernelType = Saiga::Kernel::BAPosePointMono<T>;
    KernelType::PoseJacobiType JrowPose;
    KernelType::PointJacobiType JrowPoint;
    KernelType::ResidualType res;



    b.setZero();
    A.u.setZero();
    A.v.setZero();



    SAIGA_ASSERT(A.w.IsRowMajor);


    double newChi2 = 0;
    {
        int k = 0;
        for (int i = 0; i < (int)validImages.size(); ++i)
        {
            int imgid = validImages[i];
            auto& img = scene.images[imgid];
            //            auto& extr   = scene.extrinsics[img.extr].se3;
            auto& extr   = x_u[i];
            auto& extr2  = scene.extrinsics[img.extr];
            auto& camera = scene.intrinsics[img.intr];
            StereoCamera4 scam(camera, scene.bf);

            for (auto& ip : img.stereoPoints)
            {
                if (ip.wp == -1) continue;
                if (ip.outlier)
                {
                    A.w.valuePtr()[k].get().setZero();
                    ++k;
                    continue;
                }
                BlockBAScalar w = ip.weight * img.imageWeight * scene.scale();
                int j           = pointToValidMap[ip.wp];


                //                auto& wp        = scene.worldPoints[ip.wp].p;
                auto& wp = x_v[j];

                WElem targetPosePoint;
                auto& targetPosePose   = A.u.diagonal()(i).get();
                auto& targetPointPoint = A.v.diagonal()(j).get();
                auto& targetPoseRes    = b.u(i).get();
                auto& targetPointRes   = b.v(j).get();

                if (ip.depth > 0)
                {
                    using KernelType = Saiga::Kernel::BAPosePointStereo<T>;
                    KernelType::PoseJacobiType JrowPose;
                    KernelType::PointJacobiType JrowPoint;
                    KernelType::ResidualType res;

                    KernelType::evaluateResidualAndJacobian(scam, extr, wp, ip.point, ip.depth, 1, res, JrowPose,
                                                            JrowPoint);
                    if (extr2.constant) JrowPose.setZero();


                    auto c      = res.squaredNorm();
                    auto sqrtrw = sqrt(w);
                    if (baOptions.huberStereo > 0)
                    {
                        auto rw = Kernel::huberWeight<T>(baOptions.huberStereo, c);
                        sqrtrw *= sqrt(rw(1));
                    }
                    JrowPose *= sqrtrw;
                    JrowPoint *= sqrtrw;
                    res *= sqrtrw;

                    newChi2 += res.squaredNorm();

                    targetPosePose += JrowPose.transpose() * JrowPose;
                    targetPointPoint += JrowPoint.transpose() * JrowPoint;
                    targetPosePoint = JrowPose.transpose() * JrowPoint;
                    targetPoseRes -= JrowPose.transpose() * res;
                    targetPointRes -= JrowPoint.transpose() * res;
                }
                else
                {
                    using KernelType = Saiga::Kernel::BAPosePointMono<T>;
                    KernelType::PoseJacobiType JrowPose;
                    KernelType::PointJacobiType JrowPoint;
                    KernelType::ResidualType res;

                    KernelType::evaluateResidualAndJacobian(camera, extr, wp, ip.point, 1, res, JrowPose, JrowPoint);
                    if (extr2.constant) JrowPose.setZero();

                    auto c      = res.squaredNorm();
                    auto sqrtrw = sqrt(w);
                    if (baOptions.huberMono > 0)
                    {
                        auto rw = Kernel::huberWeight<T>(baOptions.huberMono, c);
                        sqrtrw *= sqrt(rw(1));
                    }
                    JrowPose *= sqrtrw;
                    JrowPoint *= sqrtrw;
                    res *= sqrtrw;

                    newChi2 += res.squaredNorm();

                    targetPosePose += JrowPose.transpose() * JrowPose;
                    targetPointPoint += JrowPoint.transpose() * JrowPoint;
                    targetPosePoint = JrowPose.transpose() * JrowPoint;
                    targetPoseRes -= JrowPose.transpose() * res;
                    targetPointRes -= JrowPoint.transpose() * res;
                }

                A.w.innerIndexPtr()[k] = j;
                A.w.valuePtr()[k]      = targetPosePoint;

                ++k;
            }
        }
    }


    return newChi2;
}

void BARec::addDelta()
{
    oldx_u = x_u;
    oldx_v = x_v;

    for (int i = 0; i < n; ++i)
    {
        auto t = delta_x.u(i).get();
        x_u[i] = SE3::exp(t) * x_u[i];
    }
    for (int i = 0; i < m; ++i)
    {
        auto t = delta_x.v(i).get();
        x_v[i] += t;
    }
}

void BARec::revertDelta()
{
    x_u = oldx_u;
    x_v = oldx_v;
}
void BARec::finalize()
{
    Scene& scene = *_scene;

    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && optimizationOptions.debugOutput);
    for (size_t i = 0; i < validImages.size(); ++i)
    {
        auto id    = validImages[i];
        auto& extr = scene.extrinsics[id];
        if (!extr.constant) extr.se3 = x_u[i];
    }

    for (size_t i = 0; i < validPoints.size(); ++i)
    {
        Eigen::Matrix<BlockBAScalar, 3, 1> t;
        auto id = validPoints[i];
        auto& p = scene.worldPoints[id].p;
        p       = x_v[i];
    }
}


void BARec::addLambda(double lambda)
{
    applyLMDiagonal(A.u, lambda);
    applyLMDiagonal(A.v, lambda);
}



void BARec::solveLinearSystem()
{
    using namespace Eigen::Recursive;
    LinearSolverOptions loptions;
    loptions.maxIterativeIterations = optimizationOptions.maxIterativeIterations;
    loptions.iterativeTolerance     = optimizationOptions.iterativeTolerance;

    loptions.solverType = (optimizationOptions.solverType == OptimizationOptions::SolverType::Direct)
                              ? LinearSolverOptions::SolverType::Direct
                              : LinearSolverOptions::SolverType::Iterative;
    loptions.buildExplizitSchur = explizitSchur;

    solver.solve(A, delta_x, b, loptions);
}

double BARec::computeCost()
{
    Scene& scene = *_scene;

    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && optimizationOptions.debugOutput);

    using T = BlockBAScalar;

    double newChi2 = 0;
    {
        for (int i = 0; i < (int)validImages.size(); ++i)
        {
            int imgid = validImages[i];
            auto& img = scene.images[imgid];
            //            auto& extr   = scene.extrinsics[img.extr].se3;
            auto& extr   = x_u[i];
            auto& camera = scene.intrinsics[img.intr];
            StereoCamera4 scam(camera, scene.bf);

            for (auto& ip : img.stereoPoints)
            {
                if (!ip) continue;
                BlockBAScalar w = ip.weight * img.imageWeight * scene.scale();
                int j           = pointToValidMap[ip.wp];
                auto& wp        = x_v[j];

                if (ip.depth > 0)
                {
                    using KernelType = Saiga::Kernel::BAPosePointStereo<T>;
                    KernelType::ResidualType res;
                    res    = KernelType::evaluateResidual(scam, extr, wp, ip.point, ip.depth, 1);
                    auto c = res.squaredNorm();
                    if (baOptions.huberStereo > 0)
                    {
                        auto rw     = Kernel::huberWeight<T>(baOptions.huberStereo, c);
                        auto sqrtrw = sqrt(rw(1)) * sqrt(w);
                        res *= sqrtrw;
                    }
                    newChi2 += c;
                }
                else
                {
                    using KernelType = Saiga::Kernel::BAPosePointMono<T>;
                    KernelType::PoseJacobiType JrowPose;
                    KernelType::PointJacobiType JrowPoint;
                    KernelType::ResidualType res;

                    res    = KernelType::evaluateResidual(camera, extr, wp, ip.point, 1);
                    auto c = res.squaredNorm();
                    if (baOptions.huberMono > 0)
                    {
                        auto rw     = Kernel::huberWeight<T>(baOptions.huberMono, c);
                        auto sqrtrw = sqrt(rw(1)) * sqrt(w);
                        res *= sqrtrw;
                    }
                    newChi2 += c;
                }
            }
        }
    }
    return newChi2;
}


#if 0
OptimizationResults BARec::solve()
{
    Scene& scene = *_scene;

    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && optimizationOptions.debugOutput);


    initStructure(scene);

    // Set a pretty high error, so the first iteratioin doesn't fail

    float linearSolverTime = 0;
    // ======================== Variables ========================
    for (int k = 0; k < optimizationOptions.maxIterations; ++k)
    {
        if (!computeUVW(scene)) computeUVW(scene);


        LinearSolverOptions loptions;
        loptions.maxIterativeIterations = optimizationOptions.maxIterativeIterations;
        loptions.iterativeTolerance     = optimizationOptions.iterativeTolerance;

        loptions.solverType = (optimizationOptions.solverType == OptimizationOptions::SolverType::Direct)
                                  ? LinearSolverOptions::SolverType::Direct
                                  : LinearSolverOptions::SolverType::Iterative;
        loptions.buildExplizitSchur = explizitSchur;

        float t = 0;
        {
            Saiga::ScopedTimer<float> timer(t);
            solver.solve(A, delta_x, b, loptions);
        }
        linearSolverTime += t;

        //        std::cout << expand(A.u).transpose() << std::endl;
        //        std::cout << expand(A.v).transpose() << std::endl;
        //        std::cout << expand(A.w).transpose() << std::endl;
        //        std::cout << expand(b.u).transpose() << std::endl;
        //        std::cout << expand(b.v).transpose() << std::endl;
        //        std::cout << expand(delta_x.u).transpose() << std::endl;
        //        std::cout << expand(delta_x.v).transpose() << std::endl;

        plus();

        updateScene(scene);
    }
    std::cout << "totalLinearSolverTime: " << linearSolverTime << std::endl;

    // revert last step
    double finalChi2 = scene.chi2();
    if (finalChi2 > chi2)
    {
        x_u = oldx_u;
        x_v = oldx_v;
        updateScene(scene);
    }

    result.linear_solver_time = linearSolverTime;
    result.cost_final         = finalChi2;

    return result;
}
#endif



#if 0

template <typename T, typename G>
inline auto computeDerivatives(T t, G g)
{
}

template <typename T>
inline auto initializeSparseStructure(T t)
{
}
template <typename T, typename G, typename H>
inline auto solve(T t, G g, H h)
{
}


static void compactSolve()
{
    using namespace Eigen;

    using BAMatrix = SymmetricMixedMatrix22<DiagonalMatrix<MatrixScalar<Matrix<double, 6, 6>>, -1>,
                                            DiagonalMatrix<MatrixScalar<Matrix<double, 3, 3>>, -1>,
                                            SparseMatrix<MatrixScalar<Matrix<double, 6, 3>>, RowMajor>,
                                            SparseMatrix<MatrixScalar<Matrix<double, 3, 6>>, RowMajor>>;

    using BAVector = MixedVector2<Matrix<MatrixScalar<Matrix<double, 6, 1>>, -1, 1>,
                                  Matrix<MatrixScalar<Matrix<double, 3, 1>>, -1, 1>>;

    BAMatrix A;
    BAVector x, b;
    MixedSymmetricRecursiveSolver<BAMatrix, BAVector> solver;

    initializeSparseStructure(A);
    for (int k = 0; k < 10; ++k)
    {
        computeDerivatives(A, b);
        solver.solve(A, x, b);
    }
}

template<typename A, typename B, typename C, typename D>
class MixedMatrix22{
};
template<typename A>
class DiagonalMatrix{
};

static void compactSolve2()
{
    using namespace Eigen;

    // clang-format off

    using BAMatrix = SymmetricMixedMatrix2<
                        DiagonalMatrix<Matrix<double, 6, 6>, -1>,
                        DiagonalMatrix<Matrix<double, 3, 3>, -1>,
                        SparseMatrix<Matrix<double, 6, 3>, RowMajor>>;

    using BAVector = MixedVector2<
                        Matrix<Matrix<double, 6, 1>, -1, 1>,
                        Matrix<Matrix<double, 3, 1>, -1, 1>>;
    BAMatrix A;
    BAVector x, b;
    for (int k = 0; k < 10; ++k)
    {
        computeDerivatives(A, b);
        solve(A, x, b);
    }

    using BAMatrix = MixedMatrix22<
        DiagonalMatrix<Matrix<double, 6, 6>>,
        SparseMatrix  <Matrix<double, 6, 3>>,
        SparseMatrix  <Matrix<double, 3, 6>>,
        DiagonalMatrix<Matrix<double, 3, 3>>>;

    // clang-format on
}

#endif


}  // namespace Saiga
