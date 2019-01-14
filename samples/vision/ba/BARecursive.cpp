#include "BARecursive.h"

#include "saiga/imgui/imgui.h"
#include "saiga/time/timer.h"
#include "saiga/util/Algorithm.h"
#include "saiga/vision/SparseHelper.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/kernels/BAPose.h"
#include "saiga/vision/kernels/BAPosePoint.h"
#include "saiga/vision/recursiveMatrices/SparseCholesky.h"
#include "saiga/vision/recursiveMatrices/SparseInnerProduct.h"

#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"

#include <fstream>
#include <numeric>

#define NO_CG_SPEZIALIZATIONS
#define NO_CG_TYPES
using Scalar = double;
const int bn = 6;
const int bm = 6;
using Block  = Eigen::Matrix<Scalar, bn, bm>;
using Vector = Eigen::Matrix<Scalar, bn, 1>;

#include "saiga/vision/recursiveMatrices/CG.h"


namespace Saiga
{
void BARec::initStructure(Scene& scene)
{
    SAIGA_BLOCK_TIMER();
    auto imageIds   = scene.validImages();
    auto numCameras = imageIds.size();
    auto numPoints  = scene.worldPoints.size();

    n = numCameras;
    m = numPoints;

    U.resize(n);
    V.resize(m);
    W.resize(n, m);
    WT.resize(m, n);

    da.resize(n);
    db.resize(m);

    ea.resize(n);
    eb.resize(m);
    q.resize(m);

    // ==

    // tmp variables
    Vinv.resize(m);
    Y.resize(n, m);
    S.resize(n, n);
    Sdiag.resize(n);
    ej.resize(n);



    cameraPointCounts.resize(n, 0);
    cameraPointCountsScan.resize(n);
    pointCameraCounts.resize(m, 0);
    pointCameraCountsScan.resize(m);
    observations = 0;
    for (auto imgid : imageIds)
    {
        auto& img = scene.images[imgid];
        int i     = imgid;
        for (auto& ip : img.monoPoints)
        {
            if (!ip)
            {
                continue;
            }
            int j = ip.wp;
            cameraPointCounts[i]++;
            pointCameraCounts[j]++;
            observations++;
        }
    }

    exclusive_scan(cameraPointCounts.begin(), cameraPointCounts.end(), cameraPointCountsScan.begin(), 0);
    exclusive_scan(pointCameraCounts.begin(), pointCameraCounts.end(), pointCameraCountsScan.begin(), 0);



#if 0
    schurStructure.resize(n, std::vector<int>(n, -1));
    for (auto& wp : scene.worldPoints)
    {
        for (auto& ref : wp.monoreferences)
        {
            for (auto& ref2 : wp.monoreferences)
            {
                schurStructure[ref.first][ref2.first] = ref2.first;
                schurStructure[ref2.first][ref.first] = ref.first;
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
#endif

    cout << "." << endl;
    cout << "Structure Analyzed." << endl;
    cout << "Cameras: " << numCameras << endl;
    cout << "Points: " << numPoints << endl;
    cout << "Observations: " << observations << endl;
#if 0
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



void BARec::computeUVW(Scene& scene)
{
    SAIGA_BLOCK_TIMER();

    using T          = double;
    using KernelType = Saiga::Kernel::BAPosePointMono<T>;
    KernelType::PoseJacobiType JrowPose;
    KernelType::PointJacobiType JrowPoint;
    KernelType::ResidualType res;


    auto imageIds = scene.validImages();

    eb.setZero();
    U.setZero();
    ea.setZero();
    V.setZero();

    std::vector<Eigen::Triplet<WElem>> ws1;
    ws1.reserve(observations);
    W.reserve(observations);
    W.setZero();


    bool useWT = computeWT || explizitSchur || (!iterativeSolver);
    std::vector<Eigen::Triplet<WTElem>> ws2;
    if (useWT)
    {
        ws2.reserve(observations);
        WT.reserve(observations);
        WT.setZero();
    }


    {
        SAIGA_BLOCK_TIMER();
        for (auto imgid : imageIds)
        {
            auto& img    = scene.images[imgid];
            auto& extr   = scene.extrinsics[img.extr].se3;
            auto& camera = scene.intrinsics[img.intr];
            int i        = img.extr;

            SAIGA_ASSERT(i == imgid);

            if (W.IsRowMajor)
            {
                W.startVec(i);
            }

            for (auto& ip : img.monoPoints)
            {
                if (!ip)
                {
                    cout << imgid << " " << ip.wp << " " << ip.point.transpose() << endl;
                    //                                        SAIGA_ASSERT(0);
                    continue;
                }

                auto wp = scene.worldPoints[ip.wp].p;


                int j    = ip.wp;
                double w = ip.weight * scene.scale();



                KernelType::evaluateResidualAndJacobian(camera, extr, wp, ip.point, w, res, JrowPose, JrowPoint);


                U.diagonal()(i).get() += (JrowPose.transpose() * JrowPose);
                V.diagonal()(j).get() += (JrowPoint.transpose() * JrowPoint);


                WElem m = JrowPose.transpose() * JrowPoint;
                ws1.emplace_back(i, j, m);

                if (useWT)
                {
                    ws2.emplace_back(j, i, m.transpose());
                }


                if (W.IsRowMajor)
                {
                    W.insertBackByOuterInner(i, j) = m;
                }
                else
                {
                    SAIGA_ASSERT(0);
                }


                ea(i).get() += (JrowPose.transpose() * res);
                eb(j).get() += JrowPoint.transpose() * res;
            }
        }
        W.finalize();
    }

    {
        SAIGA_BLOCK_TIMER();
        //        W.setFromTriplets(ws1.begin(), ws1.end());
        if (useWT)
        {
            WT.setFromTriplets(ws2.begin(), ws2.end());
        }
    }



    double lambda = 1.0 / (scene.scale() * scene.scale());
    //        lambda        = 1;

    for (int i = 0; i < n; ++i)
    {
        U.diagonal()(i).get() += KernelType::PoseDiaBlockType::Identity() * lambda;
    }
    for (int i = 0; i < m; ++i)
    {
        V.diagonal()(i).get() += KernelType::PointDiaBlockType::Identity() * lambda;
    }
}



void BARec::solve(Scene& scene, int its)
{
    initStructure(scene);


    SAIGA_BLOCK_TIMER();


    // ======================== Variables ========================

    auto imageIds = scene.validImages();

    for (int k = 0; k < its; ++k)
    {
        computeUVW(scene);


#if 0
        cout << expand(W) << endl << endl;
        cout << expand(U.toDenseMatrix()) << endl << endl;
        cout << expand(V.toDenseMatrix()) << endl << endl;
#endif

        {
            SAIGA_BLOCK_TIMER();
            // Schur complement solution

            // Step 1 ~ 0.5%
            // Invert V
            for (int i = 0; i < m; ++i) Vinv.diagonal()(i) = V.diagonal()(i).get().inverse();
        }

        {
            SAIGA_BLOCK_TIMER();
            // Step 2
            // Compute Y ~7.74%
            Y = multSparseDiag(W, Vinv);
        }


        {
            SAIGA_BLOCK_TIMER();
            // Step 3
            // Compute the Schur complement S
            // Not sure how good the sparse matrix mult is of eigen
            // maybe own implementation because the structure is well known before hand
            // ~ 22.3 %
            // TODO: this line doesn't seem to compile with every eigen version
            //            S.resize(W.rows(), W.rows());
            //            S.reserve(schurEdges);
            //            S            = Y * WT;
            //            S            = -S;
            //            S.diagonal() = U.diagonal() + S.diagonal();
            //            cout << "S non zeros " << S.nonZeros() << endl;
            if (explizitSchur || !iterativeSolver)
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
        }
        {
            SAIGA_BLOCK_TIMER();
            // Step 4
            // Compute the right hand side of the schur system ej
            // S * da = ej
            // ~ 0.7%
            ej = ea + -(Y * eb);
        }
#if 0
        {
            // currently around of a factor 3 slower then the eigen ldlt
            SAIGA_BLOCK_TIMER();
            SparseLDLT<decltype(S), decltype(ej)> ldlt;
            ldlt.compute(S);
            da = ldlt.solve(ej);
        }
#endif

        if (!iterativeSolver)
        {
            Eigen::SparseMatrix<double> ssparse(n * asize, n * asize);
            {
                SAIGA_BLOCK_TIMER();
                // Step 5
                // Solve the schur system for da
                // ~ 5.04%

                auto triplets = sparseBlockToTriplets(S);

                ssparse.setFromTriplets(triplets.begin(), triplets.end());
            }
            {
                SAIGA_BLOCK_TIMER();

                //~61%

                Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
                //        Eigen::SimplicialLDLT<SType> solver;
                solver.compute(ssparse);
                Eigen::Matrix<double, -1, 1> deltaA = solver.solve(blockVectorToVector(ej));

                //        cout << "deltaA" << endl << deltaA << endl;

                // copy back into da
                for (int i = 0; i < n; ++i)
                {
                    da(i) = deltaA.segment(i * asize, asize);
                }
            }
        }
        else
        {
            // this CG solver is super fast :)
            SAIGA_BLOCK_TIMER();
            da.setZero();
            RecursiveDiagonalPreconditioner<MatrixScalar<Block>> P;
            Eigen::Index iters = 50;
            Scalar tol         = 1e-50;

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
            cout << "error " << tol << " iterations " << iters << endl;
        }



        {
            SAIGA_BLOCK_TIMER();
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
        }
        {
            SAIGA_BLOCK_TIMER();
            // Step 7
            // Solve the remaining partial system with the precomputed inverse of V
            /// ~0.2%
            db = multDiagVector(Vinv, q);

#if 0
            // compute residual of linear system
            auto xa                           = blockVectorToVector(da);
            auto xb                           = blockVectorToVector(db);
            Eigen::Matrix<double, -1, 1> res1 = blockMatrixToMatrix(U.toDenseMatrix()) * xa +
                                                blockMatrixToMatrix(W.toDense()) * xb - blockVectorToVector(ea);
            Eigen::Matrix<double, -1, 1> res2 = blockMatrixToMatrix(WT.toDense()) * xa +
                                                blockMatrixToMatrix(V.toDenseMatrix()) * xb - blockVectorToVector(eb);
            cout << "Error: " << res1.norm() << " " << res2.norm() << endl;
#endif
            //        cout << "da" << endl << blockVectorToVector(da).transpose() << endl;
            //        cout << "db" << endl << blockVectorToVector(db).transpose() << endl;
        }



#if 0
        // ======================== Dense Simple Solution (only for checking the correctness) ========================
        Eigen::VectorXd x1, x2;
        x1 = blockVectorToVector(da);
        x2 = blockVectorToVector(db);
        {
            SAIGA_BLOCK_TIMER();
            n *= asize;
            m *= bsize;

            // Build the complete system matrix
            Eigen::MatrixXd M(m + n, m + n);
            M.block(0, 0, n, n) = blockDiagonalToMatrix(U);
            M.block(n, n, m, m) = blockDiagonalToMatrix(V);
            M.block(0, n, n, m) = blockMatrixToMatrix(W.toDense());
            M.block(n, 0, m, n) = blockMatrixToMatrix(W.toDense()).transpose();

            // stack right hand side
            Eigen::VectorXd b(m + n);
            b.segment(0, n) = blockVectorToVector(ea);
            b.segment(n, m) = blockVectorToVector(eb);

            // compute solution
            Eigen::VectorXd x = M.ldlt().solve(b);

            double error = (M * x - b).norm();
            cout << x.transpose() << endl;
            cout << "Dense error " << error << endl;

            x1 = x.segment(0, n);
            x2 = x.segment(n, m);

            n /= asize;
            m /= bsize;
        }
#endif

        for (size_t i = 0; i < imageIds.size(); ++i)
        {
            auto id                 = imageIds[i];
            Sophus::SE3d::Tangent t = da(i).get();
            auto& se3               = scene.extrinsics[id].se3;
            se3                     = Sophus::SE3d::exp(t) * se3;
        }

        for (int i = 0; i < m; ++i)
        {
            Vec3 t  = db(i).get();
            auto& p = scene.worldPoints[i].p;
            p += t;
        }
    }
}



void BARec::imgui()
{
    ImGui::Checkbox("iterativeSolver", &iterativeSolver);
    ImGui::Checkbox("explizitSchur", &explizitSchur);
    ImGui::Checkbox("computeWT", &computeWT);
}

}  // namespace Saiga
