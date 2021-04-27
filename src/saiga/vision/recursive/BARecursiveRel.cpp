/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "BARecursiveRel.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/Algorithm.h"
#include "saiga/core/util/Thread/omp.h"
#include "saiga/vision/kernels/BA.h"
#include "saiga/vision/kernels/PGO.h"
#include "saiga/vision/kernels/Robust.h"
#include "saiga/vision/util/HistogramImage.h"
#include "saiga/vision/util/LM.h"

#include <fstream>
#include <numeric>



#define RECURSIVE_BA_USE_TIMERS false

namespace Saiga
{
void BARecRel::reserve(int n, int m)
{
    validImages.reserve(n);
    validPoints.reserve(m);
    pointToValidMap.resize(m);
    cameraPointCounts.reserve(n);
    cameraPointCountsScan.reserve(n);
    pointCameraCounts.reserve(m);
    pointCameraCountsScan.reserve(m);

    pointDiagTemp.reserve(m);
    pointResTemp.reserve(m);

    localChi2.reserve(64);

    x_u.reserve(n);
    oldx_u.reserve(n);
    x_v.reserve(n);
    oldx_v.reserve(n);
}

void BARecRel::init()
{
    //    OMP::setWaitPolicy(OMP::WaitPolicy::Active);
    //    threads = 4;
    //    std::cout << "Test sizes: " << sizeof(Scene) << " " << sizeof(BARec)<< " " << sizeof(BABase)<< " " <<
    //    sizeof(LMOptimizer) << std::endl; std::cout << "Test sizes2: " << sizeof(BAMatrix) << " " <<
    //    sizeof(BAVector)<< " " << sizeof(BASolver)<< " " << sizeof(AlignedVector<SE3>) << std::endl;


    //    threads      = OMP::getNumThreads();
    Scene& scene = *_scene;

    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && optimizationOptions.debugOutput);


    // currently the scene must be in a valid state

    if (optimizationOptions.debugOutput)
    {
        SAIGA_ASSERT(scene);
    }

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

    totalN    = 0;
    constantN = 0;


    camera_to_valid_map.resize(scene.images.size());

    for (int i = 0, validId = 0, nonConstantN = 0; i < (int)scene.images.size(); ++i)
    {
        auto& img = scene.images[i];

        camera_to_valid_map[i] = -1;
        if (!img)
        {
            // std::cout << "invalid image " << i << std::endl;
            continue;
        }

        ImageInfo info;
        info.sceneImageId = i;
        info.validId      = validId++;

        bool constant = img.constant;

        int realOffset = constant ? -1 : nonConstantN;

        info.variableId = realOffset;
        //        validImages.emplace_back(validId, realOffset);
        if (constant)
            constantN++;
        else
            nonConstantN++;
        totalN++;

        camera_to_valid_map[i] = validImages.size();
        validImages.push_back(info);
    }

    SAIGA_ASSERT(totalN == (int)validImages.size());

    for (int i = 0; i < (int)scene.worldPoints.size(); ++i)
    {
        auto& wp           = scene.worldPoints[i];
        pointToValidMap[i] = -1;
        if (!wp) continue;
        int validId        = validPoints.size();
        pointToValidMap[i] = validId;
        validPoints.push_back(i);
    }
    //    SAIGA_ASSERT(validPoints.size() == scene.worldPoints.size());

    n = totalN - constantN;
    m = validPoints.size();

    //    std::cout << n << " " << totalN << " " << constantN << std::endl;


    //    SAIGA_ASSERT(n > 0 && m > 0);

    //    A.resize(n, m);
    A.u.resize(n, n);
    A.w.resize(n, m);
    A.v.resize(m);
    //    U.resize(n);
    //    V.resize(m);

    delta_x.resize(n, m);
    b.resize(n, m);

    x_u.resize(totalN);
    oldx_u.resize(totalN);
    x_v.resize(m);
    oldx_v.resize(m);


    // Make a copy of the initial parameters
    for (auto&& info : validImages)
    {
        auto& img         = scene.images[info.sceneImageId];
        x_u[info.validId] = img.se3;
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

    std::vector<int> innerElements;
    for (auto&& info : validImages)
    {
        auto imgId  = info.sceneImageId;
        auto offset = info.variableId;
        //        std::cout << imgId << " " << offset << std::endl;
        if (offset == -1) continue;

        auto& img = scene.images[imgId];

        for (auto& ip : img.stereoPoints)
        {
            if (ip.wp == -1) continue;

            int j = pointToValidMap[ip.wp];
            cameraPointCounts[offset]++;
            pointCameraCounts[j]++;
            innerElements.push_back(j);
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


    for (int i = 0; i < observations; ++i)
    {
        A.w.innerIndexPtr()[i] = innerElements[i];
    }



    {
        // ==== Rel pose constraints and matrix A.u ====
        // this is copied from PGO
        scene.SortRelPoseConstraints();
        auto& S = A.u;
        S.setZero();



        int N = n;


        //        S.reserve();

        // make room for the diagonal element
        for (int i = 0; i < n; ++i)
        {
            S.outerIndexPtr()[i] = 1;
        }

        // Compute outer structure pointer
        // Note: the row is smaller than the column, which means we are in the upper right part of the matrix.
        for (auto& e : scene.rel_pose_constraints)
        {
            int valid_i = camera_to_valid_map[e.img1];
            int valid_j = camera_to_valid_map[e.img2];

            SAIGA_ASSERT(valid_i >= 0 && valid_i < (int)validImages.size());
            SAIGA_ASSERT(valid_j >= 0 && valid_j < (int)validImages.size());

            auto i = validImages[valid_i];
            auto j = validImages[valid_j];

            if (i.isConstant() || j.isConstant()) continue;

            SAIGA_ASSERT(i.variableId >= 0 && i.variableId < S.outerSize());
            //            SAIGA_ASSERT(i != j);
            //            SAIGA_ASSERT(i < j);

            N++;
            S.outerIndexPtr()[i.variableId]++;
        }

        S.reserve(N);


        for (int i = 0; i < N; ++i)
        {
            S.innerIndexPtr()[i] = -923875;
            S.valuePtr()[i]      = ADiag::Ones() * std::numeric_limits<double>::infinity();
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
        edgeOffsets.reserve(scene.rel_pose_constraints.size());
        std::vector<int> localOffsets(n, 1);
        for (int k = 0; k < (int)scene.rel_pose_constraints.size(); ++k)
        {
            edgeOffsets[k] = -1;
            auto e         = scene.rel_pose_constraints[k];

            auto id_i = camera_to_valid_map[e.img1];
            auto id_j = camera_to_valid_map[e.img2];
            if (id_i < 0 || id_j < 0) continue;

            auto i = validImages[id_i];
            auto j = validImages[id_j];

            if (i.isConstant() || j.isConstant()) continue;

            int li = localOffsets[i.variableId]++;

            int offseti = S.outerIndexPtr()[i.variableId] + li;

            S.innerIndexPtr()[offseti] = j.variableId;

            SAIGA_ASSERT(offseti < S.data().allocatedSize());

            edgeOffsets[k] = offseti;
        }

        //        std::cout << "test " << std::endl;
        //        std::cout << expand(S) << std::endl;
        //        exit(0);
    }

    //    auto constraints =


    // ===== Threading Tmps ======

    SAIGA_ASSERT(baOptions.helper_threads > 0);
    localChi2.resize(baOptions.helper_threads);
    pointDiagTemp.resize(baOptions.helper_threads - 1);
    pointResTemp.resize(baOptions.helper_threads - 1);
    for (auto& a : pointDiagTemp) a.resize(m);
    for (auto& a : pointResTemp) a.resize(m);

    SAIGA_ASSERT(baOptions.helper_threads == 1);
    SAIGA_ASSERT(baOptions.solver_threads == 1);


    // Setup the linear solver and anlyze the pattern
    loptions.maxIterativeIterations = optimizationOptions.maxIterativeIterations;
    loptions.iterativeTolerance     = optimizationOptions.iterativeTolerance;
    loptions.solverType             = (optimizationOptions.solverType == OptimizationOptions::SolverType::Direct)
                              ? Eigen::Recursive::LinearSolverOptions::SolverType::Direct
                              : Eigen::Recursive::LinearSolverOptions::SolverType::Iterative;
    loptions.buildExplizitSchur = optimizationOptions.buildExplizitSchur;

    if (baOptions.solver_threads == 1)
    {
        solver.analyzePattern(A, loptions);
    }

#if 0

    // Create sparsity histogram of the schur complement
    if (optimizationOptions.debugOutput)
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
        //        img.writeBinary("vision_ba_schur_sparsity.png");
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
#endif

    if (optimizationOptions.debugOutput)
    {
        std::cout << "." << std::endl;
        std::cout << "Structure Analyzed." << std::endl;
        std::cout << "Total Cameras: " << totalN << std::endl;
        std::cout << "Constant Cameras: " << constantN << std::endl;
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


double BARecRel::computeQuadraticForm()
{
    Scene& scene = *_scene;

    //    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && optimizationOptions.debugOutput);

    using T = BlockBAScalar;
    //    using KernelType = Saiga::Kernel::BAPosePointMono<T>;


    //    b.u.setZero();
    //    b.v.setZero();
    //    A.u.setZero();
    //    A.v.setZero();



    SAIGA_ASSERT(A.w.IsRowMajor);

    if (n == 0)
    {
        return 0;
    }


#pragma omp parallel num_threads(baOptions.helper_threads)
    {
        int tid = OMP::getThreadNum();

        double& newChi2 = localChi2[tid];
        newChi2         = 0;
        BDiag* bdiagArray;
        BRes* bresArray;

        if (tid == 0)
        {
            // thread 0 directly writes into the recursive matrix
            bdiagArray = &A.v.diagonal()(0).get();
            bresArray  = &b.v(0).get();
        }
        else
        {
            bdiagArray = pointDiagTemp[tid - 1].data();
            bresArray  = pointResTemp[tid - 1].data();
        }

        // every thread has to zero its own local copy
        for (int i = 0; i < m; ++i)
        {
            bdiagArray[i].setZero();
            bresArray[i].setZero();
        }

#pragma omp for
        for (auto valid_id = 0; valid_id < (int)validImages.size(); ++valid_id)
        {
            auto info = validImages[valid_id];

            // int imgid        = info.sceneImageId;
            int actualOffset = info.variableId;

            int k = A.w.outerIndexPtr()[actualOffset];

            bool constant = actualOffset == -1;
            //            std::cout << "img " << imgid << " " << actualOffset << " " << k << " const " << constant <<
            //            std::endl; SAIGA_ASSERT(k == A.w.outerIndexPtr()[i]);



            auto& img    = scene.images[info.sceneImageId];
            auto& extr   = x_u[info.validId];
            auto& camera = scene.intrinsics[img.intr];
            StereoCamera4 scam(camera, scene.bf);



            if (!constant)
            {
                // each thread can direclty right into A.u and b.u because
                // we parallize over images
                auto& targetPosePose = A.u.diagonal()(actualOffset).get();
                auto& targetPoseRes  = b.u(actualOffset).get();
                targetPosePose.setZero();
                targetPoseRes.setZero();
            }

            for (auto& ip : img.stereoPoints)
            {
                if (ip.wp == -1) continue;
                if (ip.outlier)
                {
                    if (!constant)
                    {
                        A.w.valuePtr()[k].get().setZero();
                        ++k;
                    }
                    continue;
                }
                BlockBAScalar w = ip.weight * scene.scale();
                int j           = pointToValidMap[ip.wp];


                //                auto& wp        = scene.worldPoints[ip.wp].p;
                auto& wp = x_v[j];

                //                WElem targetPosePoint;
                WElem& targetPosePoint = A.w.valuePtr()[k].get();

                //                BDiag& targetPointPoint = A.v.diagonal()(j).get();
                //                BRes& targetPointRes    = b.v(j).get();
                BDiag& targetPointPoint = bdiagArray[j];
                BRes& targetPointRes    = bresArray[j];

                if (ip.IsStereoOrDepth())
                {
                    auto stereo_point = ip.GetStereoPoint(scene.bf);

                    Matrix<double, 3, 6> JrowPose;
                    Matrix<double, 3, 3> JrowPoint;
                    auto [res, depth] = BundleAdjustmentStereo(scam, ip.point, stereo_point, extr, wp, w,
                                                               w * scene.stereo_weight, &JrowPose, &JrowPoint);

                    T loss_weight = 1.0;
                    auto res_2    = res.squaredNorm();
                    if (baOptions.huberStereo > 0)
                    {
                        auto rw = Kernel::HuberLoss<T>(baOptions.huberStereo, res_2);
                        //                        auto rw     = Kernel::CauchyLoss<T>(baOptions.huberStereo, res_2);
                        res_2       = rw(0);
                        loss_weight = rw(1);
                    }
                    //                    if (!valid_depth) loss_weight = 0;


                    newChi2 += res_2;

                    if (!constant)
                    {
                        auto& targetPosePose = A.u.diagonal()(actualOffset).get();
                        auto& targetPoseRes  = b.u(actualOffset).get();
                        targetPosePose += loss_weight * JrowPose.transpose() * JrowPose;
                        targetPosePoint = loss_weight * JrowPose.transpose() * JrowPoint;
                        targetPoseRes -= loss_weight * JrowPose.transpose() * res;
                    }
                    targetPointPoint += loss_weight * JrowPoint.transpose() * JrowPoint;
                    targetPointRes -= loss_weight * JrowPoint.transpose() * res;
                }
                else
                {
                    Matrix<double, 2, 6> JrowPose;
                    Matrix<double, 2, 3> JrowPoint;
                    auto [res, depth] = BundleAdjustment(camera, ip.point, extr, wp, w, &JrowPose, &JrowPoint);

                    T loss_weight = 1.0;
                    auto res_2    = res.squaredNorm();
                    if (baOptions.huberMono > 0)
                    {
                        auto rw = Kernel::HuberLoss<T>(baOptions.huberMono, res_2);
                        //                        auto rw     = Kernel::CauchyLoss<T>(baOptions.huberMono, res_2);
                        res_2       = rw(0);
                        loss_weight = rw(1);
                    }
                    newChi2 += res_2;

                    //                    if (!valid_depth) loss_weight = 0;
                    if (!constant)
                    {
                        auto& targetPosePose = A.u.diagonal()(actualOffset).get();
                        auto& targetPoseRes  = b.u(actualOffset).get();
                        targetPosePose += loss_weight * JrowPose.transpose() * JrowPose;
                        targetPosePoint = loss_weight * JrowPose.transpose() * JrowPoint;
                        targetPoseRes -= loss_weight * JrowPose.transpose() * res;
                    }
                    targetPointPoint += loss_weight * JrowPoint.transpose() * JrowPoint;
                    targetPointRes -= loss_weight * JrowPoint.transpose() * res;
                }

                if (!constant)
                {
                    SAIGA_ASSERT(A.w.innerIndexPtr()[k] == j);
                    ++k;
                }
            }
        }

#pragma omp for
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < baOptions.helper_threads - 1; ++j)
            {
                A.v.diagonal()(i).get() += pointDiagTemp[j][i];
                b.v(i).get() += pointResTemp[j][i];
            }
        }
    }


    double chi2_rel = 0;
    for (int c = 0; c < (int)scene.rel_pose_constraints.size(); ++c)
    {
        auto e = scene.rel_pose_constraints[c];


        auto id_i = camera_to_valid_map[e.img1];
        auto id_j = camera_to_valid_map[e.img2];
        if (id_i < 0 || id_j < 0) continue;

        auto i = validImages[id_i];
        auto j = validImages[id_j];

        Eigen::Matrix<double, 6, 6> Jrowi, Jrowj;
        Vec6 res = relPoseErrorView(e.rel_pose.inverse(), x_u[i.validId], x_u[j.validId], e.weight_rotation,
                                    e.weight_translation, &Jrowi, &Jrowj);


        if (!i.isConstant())
        {
            auto& target_ii = A.u.valuePtr()[A.u.outerIndexPtr()[i.variableId]].get();
            auto& target_ir = b.u(i.variableId).get();
            target_ii += Jrowi.transpose() * Jrowi;
            target_ir -= Jrowi.transpose() * res;
        }

        if (!j.isConstant())
        {
            auto& target_jj = A.u.valuePtr()[A.u.outerIndexPtr()[j.variableId]].get();
            auto& target_jr = b.u(j.variableId).get();
            target_jj += Jrowj.transpose() * Jrowj;
            target_jr -= Jrowj.transpose() * res;
        }

        if (!i.isConstant() && !j.isConstant())
        {
            // std::cout << "add edge " << i.sceneImageId << " " << j.sceneImageId << " " << offset << std::endl;
            SAIGA_ASSERT(edgeOffsets[c] != -1);
            auto& offset    = edgeOffsets[c];
            auto& target_ij = A.u.valuePtr()[offset].get();
            target_ij       = Jrowi.transpose() * Jrowj;
        }

        chi2_rel += res.squaredNorm();
    }


    //    std::cout << expand(A.u) << std::endl;
    //    exit(0);



    chi2_sum = chi2_rel;
    for (int i = 0; i < baOptions.helper_threads; ++i)
    {
        chi2_sum += localChi2[i];
    }


    return chi2_sum;
}

bool BARecRel::addDelta()
{
    //#pragma omp parallel num_threads(baOptions.helper_threads)
    {
#pragma omp for nowait
        for (auto valid_id = 0; valid_id < (int)validImages.size(); ++valid_id)
        {
            auto info = validImages[valid_id];
            if (info.isConstant()) continue;

            auto id     = info.validId;
            auto offset = info.variableId;
            oldx_u[id]  = x_u[id];



            Vec6 t = delta_x.u(offset).get();

            x_u[id] = Sophus::se3_expd(t) * x_u[id];

            //        Sophus::decoupled_inc(t, x_u[id]);
            //        x_u[id].translation() += t.head<3>();
            //        x_u[id].so3() = Sophus::SO3d::exp(t.tail<3>()) * x_u[id].so3();
        }

#pragma omp for
        for (int i = 0; i < m; ++i)
        {
            oldx_v[i] = x_v[i];
            Vec3 t    = delta_x.v(i).get();
            //            std::cout << i << " dp " << t.transpose() << std::endl;
            x_v[i] += t;
        }
    }
    return true;
}

void BARecRel::revertDelta()
{
    //#pragma omp parallel num_threads(threads)
    //#pragma omp parallel num_threads(baOptions.helper_threads)
    {
        //#pragma omp for nowait
        //        for (int i = 0; i < x_u.size(); ++i)
#pragma omp for
        for (auto valid_id = 0; valid_id < (int)validImages.size(); ++valid_id)
        {
            auto info = validImages[valid_id];
            if (info.isConstant()) continue;

            x_u[info.validId] = oldx_u[info.validId];
        }
#pragma omp for nowait
        for (int i = 0; i < (int)x_v.size(); ++i)
        {
            x_v[i] = oldx_v[i];
        }
    }
    //    x_u = oldx_u;
    //    x_v = oldx_v;
}
void BARecRel::finalize()
{
    Scene& scene = *_scene;

    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && optimizationOptions.debugOutput);

    //#pragma omp parallel num_threads(threads)
    //#pragma omp parallel num_threads(baOptions.helper_threads)
    {
#pragma omp for nowait
        for (auto valid_id = 0; valid_id < (int)validImages.size(); ++valid_id)
        {
            auto info = validImages[valid_id];
            if (info.isConstant()) continue;

            auto& extr = scene.images[info.sceneImageId];
            extr.se3   = x_u[info.validId];
        }
#pragma omp for
        for (int i = 0; i < (int)validPoints.size(); ++i)
        {
            auto id  = validPoints[i];
            auto& wp = scene.worldPoints[id];
            if (wp.constant) continue;
            SAIGA_ASSERT(wp);

            wp.p = x_v[i];
        }
    }
}


void BARecRel::addLambda(double lambda)
{
    // apply lm
    for (int i = 0; i < A.u.rows(); ++i)
    {
        //        auto& d = A.u.valuePtr()[A.u.outerIndexPtr()[i]].get();
        auto& d = A.u.diagonal()(i).get();
        applyLMDiagonalInner(d, lambda);
    }
    //    if (1 == 1)
    //    //    {
    //    applyLMDiagonal(A.u, lambda);
    applyLMDiagonal(A.v, lambda);
}



void BARecRel::solveLinearSystem()
{
    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && optimizationOptions.debugOutput);


    solver.solve(A, delta_x, b, loptions);
}

double BARecRel::computeCost()
{
    Scene& scene = *_scene;

    SAIGA_OPTIONAL_BLOCK_TIMER(RECURSIVE_BA_USE_TIMERS && optimizationOptions.debugOutput);

    using T = BlockBAScalar;

#pragma omp parallel num_threads(baOptions.helper_threads)
    {
        int tid = OMP::getThreadNum();

        double& newChi2 = localChi2[tid];
        newChi2         = 0;
#pragma omp for
        for (auto valid_id = 0; valid_id < (int)validImages.size(); ++valid_id)
        {
            auto info = validImages[valid_id];
            SAIGA_ASSERT(info);
            auto& img  = scene.images[info.sceneImageId];
            auto& extr = x_u[info.validId];
            //            auto& extr2  = scene.extrinsics[img.extr];
            //            auto& extr   = extr2.se3;
            auto& camera = scene.intrinsics[img.intr];

            StereoCamera4 scam(camera, scene.bf);

            for (auto& ip : img.stereoPoints)
            {
                if (!ip) continue;
                BlockBAScalar w = ip.weight * scene.scale();
                int j           = pointToValidMap[ip.wp];
                SAIGA_ASSERT(j >= 0);
                auto& wp = x_v[j];

                if (ip.IsStereoOrDepth())
                {
                    //                    using KernelType = Saiga::Kernel::BAPosePointStereo<T>;
                    //                    KernelType::ResidualType res;
                    auto stereo_point = ip.GetStereoPoint(scene.bf);
                    auto [res, depth] =
                        BundleAdjustmentStereo(scam, ip.point, stereo_point, extr, wp, w, w * scene.stereo_weight);
                    //                    res               = KernelType::evaluateResidual(scam, extr, wp, ip.point,
                    //                    stereo_point, w,
                    //                                                       w * scene.stereo_weight);
                    auto res_2 = res.squaredNorm();
                    if (baOptions.huberStereo > 0)
                    {
                        auto rw = Kernel::HuberLoss<T>(baOptions.huberStereo, res_2);
                        //                        auto rw = Kernel::CauchyLoss<T>(baOptions.huberStereo, res_2);
                        res_2 = rw(0);
                    }
                    newChi2 += res_2;
                }
                else
                {
                    //                    using KernelType = Saiga::Kernel::BAPosePointMono<T>;
                    //                    KernelType::ResidualType res;
                    //                    res        = KernelType::evaluateResidual(camera, extr, wp, ip.point, w);

                    auto [res, depth] = BundleAdjustment(scam, ip.point, extr, wp, w);

                    auto res_2 = res.squaredNorm();
                    if (baOptions.huberMono > 0)
                    {
                        auto rw = Kernel::HuberLoss<T>(baOptions.huberMono, res_2);
                        //                        auto rw = Kernel::CauchyLoss<T>(baOptions.huberMono, res_2);
                        res_2 = rw(0);
                    }
                    newChi2 += res_2;
                }
            }
        }
    }

    double chi2_rel = 0;
    for (int c = 0; c < (int)scene.rel_pose_constraints.size(); ++c)
    {
        auto rpc = scene.rel_pose_constraints[c];



        auto id_i = camera_to_valid_map[rpc.img1];
        auto id_j = camera_to_valid_map[rpc.img2];
        if (id_i < 0 || id_j < 0) continue;

        auto i = validImages[id_i];
        auto j = validImages[id_j];

        Vec6 res = relPoseErrorView(rpc.rel_pose.inverse(), x_u[i.validId], x_u[j.validId], rpc.weight_rotation,
                                    rpc.weight_translation);

        //        std::cout << "RPC BA " << rpc.img1 << " - " << rpc.img2 << " Error: " << res.squaredNorm() <<
        //        std::endl;
        chi2_rel += res.squaredNorm();
    }


    chi2_sum = chi2_rel;
    for (int i = 0; i < baOptions.helper_threads; ++i)
    {
        chi2_sum += localChi2[i];
    }


    //    finalize();
    //    std::cout << "chi2 sum " << chi2_sum << " chi2 scene " << scene.chi2() << std::endl;


    return chi2_sum;
}
}  // namespace Saiga
