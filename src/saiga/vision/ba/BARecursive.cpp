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
#include "saiga/vision/kernels/BAPose.h"
#include "saiga/vision/kernels/BAPosePoint.h"
#include "saiga/vision/kernels/Robust.h"

#include <fstream>
#include <numeric>

namespace Saiga
{
void BARec::initStructure(Scene& scene)
{
    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && options.debugOutput);

    // currently the scene must be in a valid state
    SAIGA_ASSERT(scene);

    if (options.solverType == BAOptions::SolverType::Direct)
    {
        explizitSchur = true;
        computeWT     = true;
    }
    else
    {
        explizitSchur = false;
        computeWT     = true;
    }

    //    imageIds        = scene.validImages();
    //    auto numCameras = imageIds.size();
    //    auto numPoints  = scene.worldPoints.size();


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

    U.resize(n);
    V.resize(m);

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
            if (!ip) continue;

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
    W.resize(n, m);
    W.setZero();
    W.reserve(observations);

    for (int k = 0; k < W.outerSize(); ++k)
    {
        W.outerIndexPtr()[k] = cameraPointCountsScan[k];
    }
    W.outerIndexPtr()[W.outerSize()] = observations;



    // Create sparsity histogram of the schur complement
    if (options.debugOutput)
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
    if (options.debugOutput)
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


    if (options.debugOutput)
    {
        cout << "." << endl;
        cout << "Structure Analyzed." << endl;
        cout << "Cameras: " << n << endl;
        cout << "Points: " << m << endl;
        cout << "Observations: " << observations << endl;
#if 1
        cout << "Schur Edges: " << schurEdges << endl;
        cout << "Non Zeros LSE: " << schurEdges * 6 * 6 << endl;
        cout << "Density: " << double(schurEdges * 6.0 * 6) / double(double(n) * n * 6 * 6) * 100 << "%" << endl;
#endif

#if 1
        // extended analysis
        double averageCams   = 0;
        double averagePoints = 0;
        for (auto i : cameraPointCounts) averagePoints += i;
        averagePoints /= cameraPointCounts.size();
        for (auto i : pointCameraCounts) averageCams += i;
        averageCams /= pointCameraCounts.size();
        cout << "Average Points per Camera: " << averagePoints << endl;
        cout << "Average Cameras per Point: " << averageCams << endl;
#endif
        cout << "." << endl;
    }
}



bool BARec::computeUVW(Scene& scene)
{
    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && options.debugOutput);

    using T          = BlockBAScalar;
    using KernelType = Saiga::Kernel::BAPosePointMono<T>;
    KernelType::PoseJacobiType JrowPose;
    KernelType::PointJacobiType JrowPoint;
    KernelType::ResidualType res;



    b.setZero();
    U.setZero();
    V.setZero();

    SAIGA_ASSERT(W.IsRowMajor);


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
                if (!ip) continue;
                BlockBAScalar w = ip.weight * img.imageWeight * scene.scale();
                int j           = pointToValidMap[ip.wp];
                //                auto& wp        = scene.worldPoints[ip.wp].p;
                auto& wp = x_v[j];

                WElem targetPosePoint;
                auto& targetPosePose   = U.diagonal()(i).get();
                auto& targetPointPoint = V.diagonal()(j).get();
                auto& targetPoseRes    = b.u(i).get();
                auto& targetPointRes   = b.v(j).get();

                if (ip.depth > 0)
                {
                    using KernelType = Saiga::Kernel::BAPosePointStereo<T>;
                    KernelType::PoseJacobiType JrowPose;
                    KernelType::PointJacobiType JrowPoint;
                    KernelType::ResidualType res;

                    KernelType::evaluateResidualAndJacobian(scam, extr, wp, ip.point, ip.depth, w, res, JrowPose,
                                                            JrowPoint);
                    if (extr2.constant) JrowPose.setZero();

                    auto c = res.squaredNorm();
                    newChi2 += c;
                    if (options.huberStereo > 0)
                    {
                        auto rw = Kernel::huberWeight<T>(options.huberStereo, c);
                        JrowPose *= rw;
                        JrowPoint *= rw;
                        res *= rw;
                    }
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

                    KernelType::evaluateResidualAndJacobian(camera, extr, wp, ip.point, w, res, JrowPose, JrowPoint);
                    if (extr2.constant) JrowPose.setZero();
                    auto c = res.squaredNorm();
                    newChi2 += c;
                    if (options.huberMono > 0)
                    {
                        auto rw = Kernel::huberWeight<T>(options.huberMono, c);
                        JrowPose *= rw;
                        JrowPoint *= rw;
                        res *= rw;
                    }

                    targetPosePose += JrowPose.transpose() * JrowPose;
                    targetPointPoint += JrowPoint.transpose() * JrowPoint;
                    targetPosePoint = JrowPose.transpose() * JrowPoint;
                    targetPoseRes -= JrowPose.transpose() * res;
                    targetPointRes -= JrowPoint.transpose() * res;
                }

                W.innerIndexPtr()[k] = j;
                W.valuePtr()[k]      = targetPosePoint;

                ++k;
            }
        }
    }

    if (newChi2 > chi2)
    {
        // it failed somehow
        // -> revert last step + increase lambda
        lambda *= 3.0;
        x_u  = oldx_u;
        x_v  = oldx_v;
        chi2 = 1e100;

        cerr << "Warning lm step failed!" << endl;
        return false;
    }
    else
    {
        // ok!
        chi2 = newChi2;
        lambda /= 2.0;
    }
    applyLMDiagonal(U, lambda);
    applyLMDiagonal(V, lambda);
    return true;
}


void BARec::updateScene(Scene& scene)
{
    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && options.debugOutput);
    for (size_t i = 0; i < validImages.size(); ++i)
    {
        Sophus::SE3<BlockBAScalar>::Tangent t;
#ifdef RECURSIVE_BA_VECTORIZE
        t = da(i).get().segment(0, 6);
#else
        t = delta_x.u(i).get();
#endif
        auto id    = validImages[i];
        auto& extr = scene.extrinsics[id];
        //        if (!extr.constant) extr.se3 = Sophus::SE3d::exp(t.cast<double>()) * extr.se3;
        if (!extr.constant) extr.se3 = x_u[i];
    }

    for (size_t i = 0; i < validPoints.size(); ++i)
    {
        Eigen::Matrix<BlockBAScalar, 3, 1> t;
#ifdef RECURSIVE_BA_VECTORIZE
        t = db(i).get().segment(0, 3);
#else
        t = delta_x.v(i).get();
#endif
        auto id = validPoints[i];
        auto& p = scene.worldPoints[id].p;
        //        p += t.cast<double>();
        p = x_v[i];
    }
}



void BARec::solve(Scene& scene, const BAOptions& options)
{
    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && options.debugOutput);
    this->options = options;
    initStructure(scene);

    // Set a pretty high error, so the first iteratioin doesn't fail
    chi2 = 1e100;
    // ======================== Variables ========================
    for (int k = 0; k < options.maxIterations; ++k)
    {
        if (!computeUVW(scene)) computeUVW(scene);


        LinearSolverOptions loptions;
        loptions.maxIterativeIterations = options.maxIterativeIterations;
        loptions.iterativeTolerance     = options.iterativeTolerance;

        //        loptions.solverType         = LinearSolverOptions::SolverType::Direct;
        loptions.buildExplizitSchur = explizitSchur;
        solver.solve(A, delta_x, b, loptions);

        //        cout << expand(A.u).transpose() << endl;
        //        cout << expand(A.v).transpose() << endl;
        //        cout << expand(A.w).transpose() << endl;
        //        cout << expand(b.u).transpose() << endl;
        //        cout << expand(b.v).transpose() << endl;
        //        cout << expand(delta_x.u).transpose() << endl;
        //        cout << expand(delta_x.v).transpose() << endl;

        plus();

        updateScene(scene);
    }

    // revert last step
    double finalChi2 = scene.chi2();
    if (finalChi2 > chi2)
    {
        x_u = oldx_u;
        x_v = oldx_v;
        updateScene(scene);
    }
}

void BARec::plus()
{
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
    MixedRecursiveSolver<BAMatrix, BAVector> solver;

    initializeSparseStructure(A);
    for (int k = 0; k < 10; ++k)
    {
        computeDerivatives(A, b);
        solver.solve(A, x, b);
    }
}


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


    // clang-format on
}

#endif


}  // namespace Saiga
