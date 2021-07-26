/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/vision/recursive/External/Cholesky/CG.h"
#include "saiga/vision/recursive/External/Core/NeutralElements.h"
#include "saiga/vision/recursive/Recursive.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
#include "numeric_derivative.h"


namespace Saiga
{
TEST(RecursiveLinearSolver, SparseBlock)
{
    Random::setSeed(903476346);
    srand(976157);
    // Symmetric positive sparse block matrix.
    // Used for example in PGO (6x6 Blocks) or ARAP (3x3 Blocks)

    using T               = double;
    const int block_size  = 6;
    int n                 = 20;
    int non_zeros_per_row = 3;

    using Block  = Eigen::Matrix<T, block_size, block_size>;
    using Vector = Eigen::Matrix<T, block_size, 1>;
    using AType  = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<Block>, Eigen::RowMajor>;
    using BType  = Eigen::Matrix<Eigen::Recursive::MatrixScalar<Vector>, -1, 1>;

    AType A(n, n);
    BType b(n);

    typedef Eigen::Triplet<Block> Trip;
    std::vector<Trip> tripletList;

    for (int i = 0; i < n; ++i)
    {
        // Diagonal Element
        Block diag = Block::Random();
        diag       = diag.selfadjointView<Eigen::Upper>();

        Vector bla = (Vector::Random() * 4 + (Vector::Ones() * 5));
        diag.diagonal() += bla;

        Trip t(i, i, diag);
        tripletList.push_back(t);

        // Vector
        b(i) = Vector::Random();
    }

    for (int i = 0; i < n; ++i)
    {
        auto indices = Random::uniqueIndices(non_zeros_per_row, n);
        for (auto j : indices)
        {
            if (j > i)
            {
                Block b = Block::Random();

                tripletList.push_back(Trip(i, j, b));
                tripletList.push_back(Trip(j, i, b.transpose()));
            }
        }
    }

    A.setFromTriplets(tripletList.begin(), tripletList.end());


    Eigen::Matrix<double, -1, -1> A_ex = expand(A);
    Eigen::Matrix<double, -1, 1> b_ex  = expand(b);

    {
        // Solve reference with ldlt
        Eigen::Matrix<double, -1, 1> x        = A_ex.ldlt().solve(b_ex);
        Eigen::Matrix<double, -1, 1> residual = A_ex * x - b_ex;
        EXPECT_LE(residual.squaredNorm(), 1e-10);
    }


    {
        // Solve reference with cg
        Eigen::ConjugateGradient<Eigen::Matrix<double, -1, -1>, Eigen::Upper, Eigen::DiagonalPreconditioner<double>> cg(
            A_ex);
        Eigen::Matrix<double, -1, 1> x        = cg.solve(b_ex);
        Eigen::Matrix<double, -1, 1> residual = A_ex * x - b_ex;
        EXPECT_LE(residual.squaredNorm(), 1e-10);
    }

    {
        // Solve recursive ldlt
        Eigen::RecursiveSimplicialLDLT<AType, Eigen::Upper> rec_ldlt;
        rec_ldlt.compute(A);
        BType x        = rec_ldlt.solve(b);
        BType residual = A * x - b;
        EXPECT_LE(expand(residual).squaredNorm(), 1e-10);
    }

    {
        // Solve recursive cg
        Eigen::Recursive::RecursiveDiagonalPreconditioner<Eigen::Recursive::MatrixScalar<Block>> P;
        P.compute(A);

        BType x(n);

        Eigen::Index iters = 50;
        double tol_error   = 1e-20;
        Eigen::Recursive::recursive_conjugate_gradient(
            [&](const BType& v, BType& result) { result = A.template selfadjointView<Eigen::Upper>() * v; }, b, x, P,
            iters, tol_error);


        BType residual = A * x - b;
        EXPECT_LE(expand(residual).squaredNorm(), 1e-10);
    }
}

TEST(RecursiveLinearSolver, BA)
{
    // Symmetric positive BA like matrix.
    //
    // | U  W |
    // | WT V |
    //
    // With U,V diagonal block and W sparse block.
    //
    static constexpr int blockSizeCamera = 6;
    static constexpr int blockSizePoint  = 3;

    int n                 = 3;
    int m                 = 10;
    int non_zeros_per_row = 3;

    using BlockBAScalar = double;

    using ADiag = Eigen::Matrix<BlockBAScalar, blockSizeCamera, blockSizeCamera, Eigen::RowMajor>;
    using BDiag = Eigen::Matrix<BlockBAScalar, blockSizePoint, blockSizePoint, Eigen::RowMajor>;
    using WElem = Eigen::Matrix<BlockBAScalar, blockSizeCamera, blockSizePoint, Eigen::RowMajor>;
    using ARes  = Eigen::Matrix<BlockBAScalar, blockSizeCamera, 1>;
    using BRes  = Eigen::Matrix<BlockBAScalar, blockSizePoint, 1>;

    // Block structured diagonal matrices
    using UType = Eigen::DiagonalMatrix<Eigen::Recursive::MatrixScalar<ADiag>, -1>;
    using VType = Eigen::DiagonalMatrix<Eigen::Recursive::MatrixScalar<BDiag>, -1>;

    // Block structured vectors
    using DAType = Eigen::Matrix<Eigen::Recursive::MatrixScalar<ARes>, -1, 1>;
    using DBType = Eigen::Matrix<Eigen::Recursive::MatrixScalar<BRes>, -1, 1>;

    // Block structured sparse matrix
    using WType = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<WElem>, Eigen::RowMajor>;


    using BAMatrix = Eigen::Recursive::SymmetricMixedMatrix2<UType, VType, WType>;
    using BAVector = Eigen::Recursive::MixedVector2<DAType, DBType>;
    using BASolver = Eigen::Recursive::MixedSymmetricRecursiveSolver<BAMatrix, BAVector>;


    BAMatrix A;
    BAVector x, b;
    BASolver solver;



    A.resize(n, m);
    x.resize(n, m);
    b.resize(n, m);


    setZero(A.u);
    setZero(A.v);
    setZero(A.w);

    setRandom(b.u);
    setRandom(b.v);



    for (int i = 0; i < n; ++i)
    {
        // Diagonal Element
        ADiag diag = ADiag::Random();
        diag       = diag.selfadjointView<Eigen::Upper>();

        ARes bla = (ARes::Random() * 4 + (ARes::Ones() * 5));
        diag.diagonal() += bla;
        A.u.diagonal()(i) = diag;
    }

    for (int i = 0; i < m; ++i)
    {
        // Diagonal Element
        BDiag diag = BDiag::Random();
        diag       = diag.selfadjointView<Eigen::Upper>();

        BRes bla = (BRes::Random() * 4 + (BRes::Ones() * 5));
        diag.diagonal() += bla;
        A.v.diagonal()(i) = diag;
    }

    typedef Eigen::Triplet<WElem> Trip;
    std::vector<Trip> tripletList;

    for (int i = 0; i < n; ++i)
    {
        auto indices = Random::uniqueIndices(non_zeros_per_row, m);
        for (auto j : indices)
        {
            WElem b = WElem::Random();

            tripletList.push_back(Trip(i, j, b));
        }
    }
    A.w.setFromTriplets(tripletList.begin(), tripletList.end());


    Eigen::Matrix<double, -1, 1> ref_x1;
    Eigen::Matrix<double, -1, 1> ref_x2;

    {
        // Expand A
        Eigen::Matrix<double, -1, -1> u_ex = expand(A.u);
        Eigen::Matrix<double, -1, -1> v_ex = expand(A.v);
        Eigen::Matrix<double, -1, -1> w_ex = expand(A.w);
        Eigen::Matrix<double, -1, -1> A_ex(u_ex.rows() + w_ex.cols(), u_ex.cols() + w_ex.cols());
        A_ex.setZero();

        A_ex.block(0, 0, u_ex.rows(), u_ex.cols())                     = u_ex;
        A_ex.block(u_ex.rows(), u_ex.cols(), v_ex.rows(), v_ex.cols()) = v_ex;
        A_ex.block(0, u_ex.cols(), w_ex.rows(), w_ex.cols())           = w_ex;
        A_ex                                                           = A_ex.selfadjointView<Eigen::Upper>();

        // Expand B
        Eigen::Matrix<double, -1, 1> b1 = expand(b.u);
        Eigen::Matrix<double, -1, 1> b2 = expand(b.v);
        Eigen::Matrix<double, -1, 1> b_ex(b1.rows() + b2.rows());
        b_ex.segment(0, b1.rows())         = b1;
        b_ex.segment(b1.rows(), b2.rows()) = b2;



        Eigen::Matrix<double, -1, 1> x_ex     = A_ex.ldlt().solve(b_ex);
        Eigen::Matrix<double, -1, 1> residual = A_ex * x_ex - b_ex;

        ref_x1 = x_ex.segment(0, b1.rows());
        ref_x2 = x_ex.segment(b1.rows(), b2.rows());
    }


    {
        setZero(x);
        Eigen::Recursive::LinearSolverOptions lops;
        lops.solverType = Eigen::Recursive::LinearSolverOptions::SolverType::Direct;
        solver.analyzePattern(A, lops);
        solver.solve(A, x, b, lops);
        ExpectCloseRelative(ref_x1, expand(x.u), 1e-10, false);
        ExpectCloseRelative(ref_x2, expand(x.v), 1e-10, false);
    }

    {
        setZero(x);
        Eigen::Recursive::LinearSolverOptions lops;
        lops.solverType             = Eigen::Recursive::LinearSolverOptions::SolverType::Iterative;
        lops.maxIterativeIterations = 200;
        lops.iterativeTolerance     = 1e-20;
        solver.analyzePattern(A, lops);
        solver.solve(A, x, b, lops);
        ExpectCloseRelative(ref_x1, expand(x.u), 1e-10, false);
        ExpectCloseRelative(ref_x2, expand(x.v), 1e-10, false);
    }
}


TEST(RecursiveLinearSolver, BARelPose)
{
    // Symmetric positive BA like matrix.
    //
    // | U  W |
    // | WT V |
    //
    // U: symmetric sparse block
    // V: symmetric diagonal block
    // W: sparse block
#if 1
    static constexpr int blockSizeCamera = 6;
    static constexpr int blockSizePoint  = 3;

    int n                 = 6;
    int m                 = 20;
    int non_zeros_per_row = 3;
#else
    // Debug
    static constexpr int blockSizeCamera = 3;
    static constexpr int blockSizePoint  = 2;

    int n                 = 4;
    int m                 = 4;
    int non_zeros_per_row = 1;
#endif

    using BlockBAScalar = double;

    using ADiag = Eigen::Matrix<BlockBAScalar, blockSizeCamera, blockSizeCamera, Eigen::RowMajor>;
    using BDiag = Eigen::Matrix<BlockBAScalar, blockSizePoint, blockSizePoint, Eigen::RowMajor>;
    using WElem = Eigen::Matrix<BlockBAScalar, blockSizeCamera, blockSizePoint, Eigen::RowMajor>;
    using ARes  = Eigen::Matrix<BlockBAScalar, blockSizeCamera, 1>;
    using BRes  = Eigen::Matrix<BlockBAScalar, blockSizePoint, 1>;


    using UType = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<ADiag>, Eigen::RowMajor>;
    using VType = Eigen::DiagonalMatrix<Eigen::Recursive::MatrixScalar<BDiag>, -1>;

    // Block structured vectors
    using DAType = Eigen::Matrix<Eigen::Recursive::MatrixScalar<ARes>, -1, 1>;
    using DBType = Eigen::Matrix<Eigen::Recursive::MatrixScalar<BRes>, -1, 1>;

    // Block structured sparse matrix
    using WType = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<WElem>, Eigen::RowMajor>;


    using BAMatrix = Eigen::Recursive::SymmetricMixedMatrix2<UType, VType, WType>;
    using BAVector = Eigen::Recursive::MixedVector2<DAType, DBType>;
    using BASolver = Eigen::Recursive::MixedSymmetricRecursiveSolver<BAMatrix, BAVector>;


    BAMatrix A;
    BAVector x, b;
    BASolver solver;



    //    A.resize(n, m);
    A.u.resize(n, n);
    A.w.resize(n, m);
    A.v.resize(m);

    x.resize(n, m);
    b.resize(n, m);


    setZero(A.u);
    setZero(A.v);
    setZero(A.w);

    setRandom(b.u);
    setRandom(b.v);



    {
        typedef Eigen::Triplet<ADiag> Trip;
        std::vector<Trip> tripletList;

        for (int i = 0; i < n; ++i)
        {
            // Diagonal Element
            ADiag diag = ADiag::Random();
            diag       = diag.selfadjointView<Eigen::Upper>();

            ARes bla = (ARes::Random() * 4 + (ARes::Ones() * 5));
            diag.diagonal() += bla;
            tripletList.push_back(Trip(i, i, diag));
        }



        for (int i = 0; i < n; ++i)
        {
            auto indices = Random::uniqueIndices(2, n);
            for (auto j : indices)
            {
                if (i < j)
                {
                    ADiag b = ADiag::Random();
                    tripletList.push_back(Trip(i, j, b));
                    //                    tripletList.push_back(Trip(j, i, b.transpose()));
                }
            }
        }
        A.u.setFromTriplets(tripletList.begin(), tripletList.end());
    }

    for (int i = 0; i < m; ++i)
    {
        // Diagonal Element
        BDiag diag = BDiag::Random();
        diag       = diag.selfadjointView<Eigen::Upper>();

        BRes bla = (BRes::Random() * 4 + (BRes::Ones() * 5));
        diag.diagonal() += bla;
        A.v.diagonal()(i) = diag;
    }

    typedef Eigen::Triplet<WElem> Trip;
    std::vector<Trip> tripletList;

    for (int i = 0; i < n; ++i)
    {
        auto indices = Random::uniqueIndices(non_zeros_per_row, m);
        for (auto j : indices)
        {
            WElem b = WElem::Random();

            tripletList.push_back(Trip(i, j, b));
        }
    }
    A.w.setFromTriplets(tripletList.begin(), tripletList.end());


    Eigen::Matrix<double, -1, 1> ref_x1;
    Eigen::Matrix<double, -1, 1> ref_x2;

    {
        // Expand A
        Eigen::Matrix<double, -1, -1> u_ex = expand(A.u);
        Eigen::Matrix<double, -1, -1> v_ex = expand(A.v);
        Eigen::Matrix<double, -1, -1> w_ex = expand(A.w);
        Eigen::Matrix<double, -1, -1> A_ex(u_ex.rows() + w_ex.cols(), u_ex.cols() + w_ex.cols());
        A_ex.setZero();

        A_ex.block(0, 0, u_ex.rows(), u_ex.cols())                     = u_ex;
        A_ex.block(u_ex.rows(), u_ex.cols(), v_ex.rows(), v_ex.cols()) = v_ex;
        A_ex.block(0, u_ex.cols(), w_ex.rows(), w_ex.cols())           = w_ex;
        A_ex                                                           = A_ex.selfadjointView<Eigen::Upper>();

        // Expand B
        Eigen::Matrix<double, -1, 1> b1 = expand(b.u);
        Eigen::Matrix<double, -1, 1> b2 = expand(b.v);
        Eigen::Matrix<double, -1, 1> b_ex(b1.rows() + b2.rows());
        b_ex.segment(0, b1.rows())         = b1;
        b_ex.segment(b1.rows(), b2.rows()) = b2;


        Eigen::Matrix<double, -1, 1> x_ex     = A_ex.ldlt().solve(b_ex);
        Eigen::Matrix<double, -1, 1> residual = A_ex * x_ex - b_ex;

        ref_x1 = x_ex.segment(0, b1.rows());
        ref_x2 = x_ex.segment(b1.rows(), b2.rows());
    }


    {
        setZero(x);
        Eigen::Recursive::LinearSolverOptions lops;
        lops.solverType = Eigen::Recursive::LinearSolverOptions::SolverType::Direct;
        lops.cholmod    = false;
        solver.analyzePattern(A, lops);
        solver.solve(A, x, b, lops);
        ExpectCloseRelative(ref_x1, expand(x.u), 1e-10, false);
        ExpectCloseRelative(ref_x2, expand(x.v), 1e-10, false);
    }

    {
        setZero(x);
        Eigen::Recursive::LinearSolverOptions lops;
        lops.solverType             = Eigen::Recursive::LinearSolverOptions::SolverType::Iterative;
        lops.maxIterativeIterations = 200;
        lops.iterativeTolerance     = 1e-20;
        lops.buildExplizitSchur     = true;
        solver.analyzePattern(A, lops);
        solver.solve(A, x, b, lops);
        ExpectCloseRelative(ref_x1, expand(x.u), 1e-10, false);
        ExpectCloseRelative(ref_x2, expand(x.v), 1e-10, false);
    }
}

}  // namespace Saiga
