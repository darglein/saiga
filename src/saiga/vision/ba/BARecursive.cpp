/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "BARecursive.h"

#include "saiga/imgui/imgui.h"
#include "saiga/time/timer.h"
#include "saiga/util/Algorithm.h"
#include "saiga/vision/HistogramImage.h"
#include "saiga/vision/SparseHelper.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/kernels/BAPose.h"
#include "saiga/vision/kernels/BAPosePoint.h"
#include "saiga/vision/kernels/Robust.h"
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
const int bn = Saiga::blockSizeCamera;
const int bm = Saiga::blockSizeCamera;
using Block  = Eigen::Matrix<Scalar, bn, bm>;
using Vector = Eigen::Matrix<Scalar, bn, 1>;

#include "saiga/vision/recursiveMatrices/CG.h"


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
        explizitSchur = true;
        computeWT     = true;
    }

    //    imageIds        = scene.validImages();
    //    auto numCameras = imageIds.size();
    //    auto numPoints  = scene.worldPoints.size();


    // Check how many valid and cameras exist and construct the compact index sets
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

    da.resize(n);
    db.resize(m);

    ea.resize(n);
    eb.resize(m);
    q.resize(m);

    // ==

    // tmp variables
    Vinv.resize(m);
    Y.resize(n, m);
    Sdiag.resize(n);
    ej.resize(n);
    S.resize(n, n);


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

    auto test1 = exclusive_scan(cameraPointCounts.begin(), cameraPointCounts.end(), cameraPointCountsScan.begin(), 0);
    auto test2 = exclusive_scan(pointCameraCounts.begin(), pointCameraCounts.end(), pointCameraCountsScan.begin(), 0);

    SAIGA_ASSERT(test1 == observations && test2 == observations);

    // preset the outer matrix structure
    W.resize(n, m);
    WT.resize(m, n);
    W.setZero();
    WT.setZero();
    W.reserve(observations);
    WT.reserve(observations);

    for (int k = 0; k < W.outerSize(); ++k)
    {
        W.outerIndexPtr()[k] = cameraPointCountsScan[k];
    }
    W.outerIndexPtr()[W.outerSize()] = observations;


    for (int k = 0; k < WT.outerSize(); ++k)
    {
        WT.outerIndexPtr()[k] = pointCameraCountsScan[k];
    }
    WT.outerIndexPtr()[WT.outerSize()] = observations;



    // Create sparsity histogram of the schur complement
    if (options.debugOutput)
    {
        HistogramImage img(n, n, 512, 512);
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



void BARec::computeUVW(Scene& scene)
{
    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && options.debugOutput);

    using T          = BlockBAScalar;
    using KernelType = Saiga::Kernel::BAPosePointMono<T>;
    KernelType::PoseJacobiType JrowPose;
    KernelType::PointJacobiType JrowPoint;
    KernelType::ResidualType res;



    eb.setZero();
    U.setZero();
    ea.setZero();
    V.setZero();

    SAIGA_ASSERT(W.IsRowMajor && WT.IsRowMajor);

    bool useWT = computeWT;


    std::vector<int> tmpPointCameraCounts(m, 0);



    {
        chi2  = 0;
        int k = 0;
        for (int i = 0; i < (int)validImages.size(); ++i)
        {
            int imgid    = validImages[i];
            auto& img    = scene.images[imgid];
            auto& extr   = scene.extrinsics[img.extr].se3;
            auto& camera = scene.intrinsics[img.intr];
            StereoCamera4 scam(camera, scene.bf);

            for (auto& ip : img.stereoPoints)
            {
                if (!ip) continue;
                auto& wp        = scene.worldPoints[ip.wp].p;
                BlockBAScalar w = ip.weight * img.imageWeight * scene.scale();
                int j           = pointToValidMap[ip.wp];

                WElem targetPosePoint;
                auto& targetPosePose   = U.diagonal()(i).get();
                auto& targetPointPoint = V.diagonal()(j).get();
                auto& targetPoseRes    = ea(i).get();
                auto& targetPointRes   = eb(j).get();

                if (ip.depth > 0)
                {
                    using KernelType = Saiga::Kernel::BAPosePointStereo<T>;
                    KernelType::PoseJacobiType JrowPose;
                    KernelType::PointJacobiType JrowPoint;
                    KernelType::ResidualType res;

                    KernelType::evaluateResidualAndJacobian(scam, extr, wp, ip.point, ip.depth, w, res, JrowPose,
                                                            JrowPoint);
                    auto c = res.squaredNorm();
                    chi2 += c;
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
                    targetPoseRes += JrowPose.transpose() * res;
                    targetPointRes += JrowPoint.transpose() * res;
                }
                else
                {
                    using KernelType = Saiga::Kernel::BAPosePointMono<T>;
                    KernelType::PoseJacobiType JrowPose;
                    KernelType::PointJacobiType JrowPoint;
                    KernelType::ResidualType res;

                    KernelType::evaluateResidualAndJacobian(camera, extr, wp, ip.point, w, res, JrowPose, JrowPoint);
                    auto c = res.squaredNorm();
                    chi2 += c;
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
                    targetPoseRes += JrowPose.transpose() * res;
                    targetPointRes += JrowPoint.transpose() * res;
                }


                // Insert into W and WT
                if (useWT)
                {
                    int x                      = tmpPointCameraCounts[j];
                    int offset                 = WT.outerIndexPtr()[j] + x;
                    WT.innerIndexPtr()[offset] = i;
                    WT.valuePtr()[offset]      = targetPosePoint.transpose();
                    tmpPointCameraCounts[j]++;
                }
                W.innerIndexPtr()[k] = j;
                W.valuePtr()[k]      = targetPosePoint;

                ++k;
            }
        }
    }


    double lambda = 1 / (scene.scale() * scene.scale());


#if 0
    lambda = 1;
    // g2o simple lambda
    for (int i = 0; i < n; ++i)
    {
        U.diagonal()(i).get() += ADiag::Identity() * lambda;
    }
    for (int i = 0; i < m; ++i)
    {
        V.diagonal()(i).get() += BDiag::Identity() * lambda;
    }
#else
    // lm lambda

    lambda = 1.0 / 1.00e+04;
    // value from ceres
    double min_lm_diagonal = 1e-6;
    double max_lm_diagonal = 1e32;

    for (int i = 0; i < n; ++i)
    {
        auto& diag = U.diagonal()(i).get();

        for (int k = 0; k < diag.RowsAtCompileTime; ++k)
        {
            auto& value = diag.diagonal()(k);
            value       = value + lambda * value;
            value       = clamp(value, min_lm_diagonal, max_lm_diagonal);
        }
    }
    for (int i = 0; i < m; ++i)
    {
        auto& diag = V.diagonal()(i).get();
        for (int k = 0; k < diag.RowsAtCompileTime; ++k)
        {
            auto& value = diag.diagonal()(k);
            value       = value + lambda * value;
            value       = clamp(value, min_lm_diagonal, max_lm_diagonal);
        }
        //        diag += BDiag::Identity() * lambda;
    }
#endif
}

void BARec::computeSchur()
{
    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && options.debugOutput);
    {
        // Schur complement solution

        // Step 1 ~ 0.5%
        // Invert V
        for (int i = 0; i < m; ++i) Vinv.diagonal()(i) = V.diagonal()(i).get().inverse();

        // Step 2
        // Compute Y ~7.74%
        Y = multSparseDiag(W, Vinv);
    }

    {
        // Step 3
        // Compute the Schur complement S
        // Not sure how good the sparse matrix mult is of eigen
        // maybe own implementation because the structure is well known before hand
        // ~ 22.3 %
        if (explizitSchur)
        {
            S            = Y * WT;
            S            = -S;
            S.diagonal() = U.diagonal() + S.diagonal();
        }
        else
        {
            diagInnerProductTransposed(Y, W, Sdiag);
            Sdiag.diagonal() = U.diagonal() - Sdiag.diagonal();
        }

        // Step 4
        // Compute the right hand side of the schur system ej
        // S * da = ej
        ej = ea + -(Y * eb);
    }
}

void BARec::solveSchur()
{
    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && options.debugOutput);
    if (options.solverType == BAOptions::SolverType::Direct)
    {
//            SAIGA_BLOCK_TIMER("LDLT");
#if 0
        {
            // currently around of a factor 3 slower then the eigen ldlt
            SAIGA_BLOCK_TIMER();
            SparseRecursiveLDLT<decltype(S), decltype(ej)> ldlt;
            ldlt.compute(S);
            da = ldlt.solve(ej);
        }

#else
        Eigen::SparseMatrix<BlockBAScalar> ssparse(n * blockSizeCamera, n * blockSizeCamera);
        {
            // Step 5
            // Solve the schur system for da
            // ~ 5.04%

            auto triplets = sparseBlockToTriplets(S);

            ssparse.setFromTriplets(triplets.begin(), triplets.end());
        }
        {
            //~61%

            Eigen::SimplicialLDLT<Eigen::SparseMatrix<BlockBAScalar>> solver;
            //        Eigen::SimplicialLDLT<SType> solver;
            solver.compute(ssparse);

            auto b                                     = expand(ej);
            Eigen::Matrix<BlockBAScalar, -1, 1> deltaA = solver.solve(b);

            //        cout << "deltaA" << endl << deltaA << endl;

            // copy back into da
            for (int i = 0; i < n; ++i)
            {
                da(i) = deltaA.segment<blockSizeCamera>(i * blockSizeCamera);
            }
        }
#endif
    }
    else
    {
        // this CG solver is super fast :)
        //            SAIGA_BLOCK_TIMER("CG");
        da.setZero();
        RecursiveDiagonalPreconditioner<MatrixScalar<Block>> P;
        Eigen::Index iters = options.maxIterativeIterations;
        Scalar tol         = options.iterativeTolerance;

        if (explizitSchur)
        {
            P.compute(S);
            DAType tmp(n);
            recursive_conjugate_gradient(
                [&](const DAType& v) {
                    tmp = S * v;
                    return tmp;
                },
                ej, da, P, iters, tol);
        }
        else
        {
            P.compute(Sdiag);
            DBType q(m);
            DAType tmp(n);
            recursive_conjugate_gradient(
                [&](const DAType& v) {
                    // x = U * p - Y * WT * p
                    if (computeWT)
                    {
                        tmp = Y * (WT * v);
                    }
                    else
                    {
                        multSparseRowTransposedVector(W, v, q);
                        tmp = Y * q;
                    }
                    tmp = (U.diagonal().array() * v.array()) - tmp.array();
                    return tmp;
                },
                ej, da, P, iters, tol);
        }
        if (options.debugOutput) cout << "error " << tol << " iterations " << iters << endl;
    }
}

void BARec::finalizeSchur()
{
    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && options.debugOutput);
    {
        //            SAIGA_BLOCK_TIMER();
        // Step 6
        // Substitute the solultion deltaA into the original system and
        // bring it to the right hand side
        // ~1.6%

        if (computeWT)
        {
            q = WT * da;
        }
        else
        {
            multSparseRowTransposedVector(W, da, q);
        }
        q = eb - q;
        //            q = eb + -WT * da;
        //        cout << "qref" << endl
        //             << (blockVectorToVector(eb) - blockMatrixToMatrix(WT.toDense()) *
        //             blockVectorToVector(da)) << endl;

        //        cout << "q" << endl << blockVectorToVector(q) << endl;

        // Step 7
        // Solve the remaining partial system with the precomputed inverse of V
        /// ~0.2%
        db = multDiagVector(Vinv, q);
    }
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
        t = da(i).get();
#endif
        auto id   = validImages[i];
        auto& se3 = scene.extrinsics[id].se3;
        se3       = Sophus::SE3d::exp(t.cast<double>()) * se3;
    }

    for (size_t i = 0; i < validPoints.size(); ++i)
    {
        Eigen::Matrix<BlockBAScalar, 3, 1> t;
#ifdef RECURSIVE_BA_VECTORIZE
        t = db(i).get().segment(0, 3);
#else
        t = db(i).get();
#endif
        auto id = validPoints[i];
        auto& p = scene.worldPoints[id].p;
        p += t.cast<double>();
    }
    Sophus::Sim3d a;
    a.Adj();
}



void BARec::solve(Scene& scene, const BAOptions& options)
{
    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && options.debugOutput);
    this->options = options;
    initStructure(scene);



    // ======================== Variables ========================


    for (int k = 0; k < options.maxIterations; ++k)
    {
        computeUVW(scene);


        if (options.debugOutput)
        {
            cout << "chi2 = " << chi2 << endl;
        }

        SAIGA_BLOCK_TIMER();
        computeSchur();

#if 0
        cout << expand(W) << endl << endl;
        cout << expand(U.toDenseMatrix()) << endl << endl;
        cout << expand(V.toDenseMatrix()) << endl << endl;
        return;
#endif

        solveSchur();
        finalizeSchur();



        updateScene(scene);
    }
}



}  // namespace Saiga
