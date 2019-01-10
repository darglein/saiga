#include "BARecursive.h"

#include "saiga/time/timer.h"
#include "saiga/vision/SparseHelper.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/kernels/BAPose.h"
#include "saiga/vision/kernels/BAPosePoint.h"
#include "saiga/vision/recursiveMatrices/SparseCholesky.h"

#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"

#include <fstream>

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

    cout << "ba with " << numCameras << " cameras and " << numPoints << " points" << endl;

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


    // ==

    // tmp variables
    Vinv.resize(m);
    Y.resize(n, m);
    S.resize(n, n);
    ej.resize(n);
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
    std::vector<Eigen::Triplet<WTElem>> ws2;

    W.reserve(225911);
    WT.reserve(225911);

    W.setZero();
    WT.setZero();

    //        for (auto& img : scene.images)
    for (auto imgid : imageIds)
    {
        auto& img    = scene.images[imgid];
        auto& extr   = scene.extrinsics[img.extr].se3;
        auto& camera = scene.intrinsics[img.intr];

        for (auto& ip : img.monoPoints)
        {
            if (!ip)
            {
                cout << imgid << " " << ip.wp << " " << ip.point.transpose() << endl;
                //                                        SAIGA_ASSERT(0);
                continue;
            }

            auto wp = scene.worldPoints[ip.wp].p;


            int i    = img.extr;
            int j    = ip.wp;
            double w = ip.weight * scene.scale();



            KernelType::evaluateResidualAndJacobian(camera, extr, wp, ip.point, w, res, JrowPose, JrowPoint);


            U.diagonal()(i).get() += (JrowPose.transpose() * JrowPose);
            V.diagonal()(j).get() += (JrowPoint.transpose() * JrowPoint);


            WElem m = JrowPose.transpose() * JrowPoint;
            ws1.emplace_back(i, j, m);
            ws2.emplace_back(j, i, m.transpose());

            //                W.setBlock(i, j, JrowPose.transpose() * JrowPoint);
            //            W.insert(i, j) = m;
            //            cout << "insert " << j << " " << i << endl;
            WT.insert(j, i) = m.transpose();


            ea(i).get() += (JrowPose.transpose() * res);
            eb(j).get() += JrowPoint.transpose() * res;
        }
    }

    {
        W.setFromTriplets(ws1.begin(), ws1.end());
        WT.setFromTriplets(ws2.begin(), ws2.end());
    }


    //        double lambda = 1.0 / (scene.intrinsics.front().fx * scene.intrinsics.front().fx);
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
            S            = -(Y * WT).eval();
            S.diagonal() = U.diagonal() + S.diagonal();
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

        {
            // currently around of a factor 3 slower then the eigen ldlt
            SAIGA_BLOCK_TIMER();
            SparseLDLT<decltype(S), decltype(ej)> ldlt;
            ldlt.compute(S);
            da = ldlt.solve(ej);
        }
#endif

        {
            // this CG solver is super fast :)
            SAIGA_BLOCK_TIMER();
            da.setZero();
            RecursiveDiagonalPreconditioner<MatrixScalar<Block>> P;
            Eigen::Index iters = 50;
            Scalar tol         = 1e-50;

            P.compute(S);
            conjugate_gradient2(S, ej, da, P, iters, tol);

            cout << "error " << tol << " iterations " << iters << endl;
        }

        DBType q;
        {
            SAIGA_BLOCK_TIMER();
            // Step 6
            // Substitute the solultion deltaA into the original system and
            // bring it to the right hand side
            // ~1.6%
            q = eb + -WT * da;
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



}  // namespace Saiga
