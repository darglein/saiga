/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/time/timer.h"
#include "saiga/util/random.h"
#include "saiga/vision/MatrixScalar.h"
#include "saiga/vision/VisionIncludes.h"

#include "Eigen/Sparse"


using namespace Saiga;

static void printVectorInstructions()
{
    cout << "Eigen Version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION
         << endl;

    std::cout << "defined EIGEN Macros:" << std::endl;

#ifdef EIGEN_NO_DEBUG
    std::cout << "EIGEN_NO_DEBUG" << std::endl;
#else
    std::cout << "EIGEN_DEBUG" << std::endl;
#endif

#ifdef EIGEN_VECTORIZE_FMA
    std::cout << "EIGEN_VECTORIZE_FMA" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_SSE3
    std::cout << "EIGEN_VECTORIZE_SSE3" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_SSSE3
    std::cout << "EIGEN_VECTORIZE_SSSE3" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_SSE4_1
    std::cout << "EIGEN_VECTORIZE_SSE4_1" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_SSE4_2
    std::cout << "EIGEN_VECTORIZE_SSE4_2" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_AVX
    std::cout << "EIGEN_VECTORIZE_AVX" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_AVX2
    std::cout << "EIGEN_VECTORIZE_AVX2" << std::endl;
#endif

    std::cout << std::endl;
}


template <typename T, int options>
std::vector<Eigen::Triplet<T>> to_triplets(const Eigen::SparseMatrix<T, options>& M)
{
    std::vector<Eigen::Triplet<T>> v;
    for (int i = 0; i < M.outerSize(); i++)
        for (typename Eigen::SparseMatrix<T, options>::InnerIterator it(M, i); it; ++it)
            v.emplace_back(it.row(), it.col(), it.value());
    return v;
}


void simpleSchurTest()
{
    // Solution of the following block-structured linear system
    //
    // | U   W |   | da |   | ea |
    // | Wt  V | * | db | = | eb |
    //
    // , where
    // U and V are diagonal matrices, and W is sparse.
    // V is assumed to be much larger then U.
    // If U is larger the schur complement should be computed in the other direction.

    // ======================== Parameters ========================

    // size of U
    int n = 12 * 6;
    // size of V
    int m = 900 * 3;
    // maximum number of non-zero elements per row in W
    int maxElementsPerRow = 35 * 3;

    using Vector = Eigen::Matrix<double, -1, 1>;
    using Matrix = Eigen::Matrix<double, -1, -1>;

    // ======================== Initialize ========================

    // Diagonal matrices U and V
    // All elements are positive!
    Eigen::DiagonalMatrix<double, -1> U(n);
    Eigen::DiagonalMatrix<double, -1> V(m);
    for (int i = 0; i < n; ++i) U.diagonal()(i) = Random::sampleDouble(0.1, 10);
    for (int i = 0; i < m; ++i) V.diagonal()(i) = Random::sampleDouble(0.1, 10);



    // Right hand side of the linear system
    Vector ea(n);
    Vector eb(m);
    ea.setRandom();
    eb.setRandom();

    Eigen::SparseMatrix<double, Eigen::RowMajor> W(n, m);
    W.reserve(n * maxElementsPerRow);
    for (int i = 0; i < n; ++i)
    {
        auto v = Random::uniqueIndices(maxElementsPerRow, m);
        for (auto j : v)
        {
            W.insert(i, j) = Random::sampleDouble(-5, 5);
        }
    }

    // ========================================================================================================

    if (n < 10)  // compute dense solution only for small problems
    {
        SAIGA_BLOCK_TIMER();
        // dense solution
        Matrix A(n + m, n + m);
        A.block(0, 0, n, n) = U.toDenseMatrix();
        A.block(n, n, m, m) = V.toDenseMatrix();
        A.block(0, n, n, m) = W.toDense();
        A.block(n, 0, m, n) = W.toDense().transpose();

        Vector e(n + m);
        e.segment(0, n) = ea;
        e.segment(n, m) = eb;

        Vector delta;
        delta = A.ldlt().solve(e);

        cout << "dense " << (A * delta - e).norm() << endl;
    }

    {
        SAIGA_BLOCK_TIMER();
        // sparse solution
        Eigen::SparseMatrix<double> A(n + m, n + m);

        std::vector<Eigen::Triplet<double>> triplets;
        for (int i = 0; i < n; ++i) triplets.emplace_back(i, i, U.diagonal()(i));
        for (int i = 0; i < m; ++i) triplets.emplace_back(i + n, i + n, V.diagonal()(i));

        auto mtri  = to_triplets(W);
        auto mtrit = to_triplets(W.transpose().eval());

        // add offsets
        for (auto& tri : mtri) tri = Eigen::Triplet<double>(tri.row(), tri.col() + n, tri.value());
        for (auto& tri : mtrit) tri = Eigen::Triplet<double>(tri.row() + n, tri.col(), tri.value());

        triplets.insert(triplets.end(), mtri.begin(), mtri.end());
        triplets.insert(triplets.end(), mtrit.begin(), mtrit.end());

        A.setFromTriplets(triplets.begin(), triplets.end());

        Vector e(n + m);
        e.segment(0, n) = ea;
        e.segment(n, m) = eb;

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        Vector delta = solver.solve(e);

        //        cout << "sparse " << delta.transpose() << endl;
        cout << "sparse " << (A * delta - e).norm() << endl;
    }



    {
        SAIGA_BLOCK_TIMER();

        // Schur complement solution

        // Step 1
        // Invert V
        Eigen::DiagonalMatrix<double, -1> Vinv(m);
        for (int i = 0; i < m; ++i) Vinv.diagonal()(i) = 1.0 / V.diagonal()(i);

        // Step 2
        // Compute Y
        Eigen::SparseMatrix<double, Eigen::RowMajor> Y(n, m);
        Y = W * Vinv;

        // Step 3
        // Compute the Schur complement S
        // Not sure how good the sparse matrix mult is of eigen
        // maybe own implementation because the structure is well known before hand
        Eigen::SparseMatrix<double> S(n, n);
        S            = -Y * W.transpose();
        S.diagonal() = U.diagonal() + S.diagonal();

        // Step 4
        // Compute the right hand side of the schur system ej
        // S * da = ej
        Vector ej(n);
        ej = ea - Y * eb;

        // Step 5
        // Solve the schur system for da
        Vector deltaA(n);
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(S);
        deltaA = solver.solve(ej);

        // Step 6
        // Substitute the solultion deltaA into the original system and
        // bring it to the right hand side
        Vector q = eb - W.transpose() * deltaA;

        // Step 7
        // Solve the remaining partial system with the precomputed inverse of V
        Vector deltaB(m);
        deltaB = Vinv * q;

        cout << "sparse schur " << (U * deltaA + W * deltaB - ea).norm() << " "
             << (W.transpose() * deltaA + V * deltaB - eb).norm() << endl;
    }
    cout << endl;
}

#define PRINT_MATRIX

void baBlockSchurTest()
{
    // Solution of the following block-structured linear system
    //
    // | U   W |   | da |   | ea |
    // | Wt  V | * | db | = | eb |
    //
    // , where
    // U and V are block diagonal matrices, and W is sparse.
    // V is assumed to be much larger then U.
    // If U is larger the schur complement should be computed in the other direction.

    // ======================== Parameters ========================

    // Block size
    const int asize = 6;
    const int bsize = 3;

    // size of U
    int n = 2;
    // size of V
    int m = 3;

    // maximum number of non-zero elements per row in W
    int maxElementsPerRow = 2;

    // ======================== Types ========================

    using T = double;

    // block types
    using ADiag = Eigen::Matrix<T, asize, asize>;
    using BDiag = Eigen::Matrix<T, bsize, bsize>;
    using WElem = Eigen::Matrix<T, asize, bsize>;
    using ARes  = Eigen::Matrix<T, asize, 1>;
    using BRes  = Eigen::Matrix<T, bsize, 1>;

    // Block structured diagonal matrices
    using UType = Eigen::DiagonalMatrix<MatrixScalar<ADiag>, -1>;
    using VType = Eigen::DiagonalMatrix<MatrixScalar<BDiag>, -1>;

    // Block structured vectors
    using DAType = Eigen::Matrix<MatrixScalar<ARes>, -1, 1>;
    using DBType = Eigen::Matrix<MatrixScalar<BRes>, -1, 1>;

    // Block structured sparse matrix
    using WType = Eigen::SparseMatrix<MatrixScalar<WElem>>;

    // ======================== Variables ========================

    UType U(n);
    VType V(m);
    WType W(n, m);

    DAType da(n);
    DBType db(m);

    DAType ea(n);
    DBType eb(m);


    // ======================== Initialize U,V,W,ea,eb ========================

    // Init U,V with random symmetric square matrices, but add a large value to the diagonal to ensure positive
    // definiteness and low condition number This is similar to the LM update

    double largeValue = 3;
    for (int i = 0; i < n; ++i)
    {
        ADiag a         = largeValue * ADiag::Identity() + ADiag::Random();
        U.diagonal()(i) = a.selfadjointView<Eigen::Upper>();
    }
    for (int i = 0; i < m; ++i)
    {
        BDiag a         = largeValue * BDiag::Identity() + BDiag::Random();
        V.diagonal()(i) = a.selfadjointView<Eigen::Upper>();
    }

#ifdef PRINT_MATRIX
    // debug print U and V
    cout << "U" << endl << blockDiagonalToMatrix(U) << endl;
    cout << "V" << endl << blockDiagonalToMatrix(V) << endl;
#endif


    // Init W with randoms blocks
    W.reserve(n * maxElementsPerRow);
    for (int i = 0; i < n; ++i)
    {
        auto v = Random::uniqueIndices(maxElementsPerRow, m);
        for (auto j : v)
        {
            W.insert(i, j) = WElem::Random();
        }
    }

#ifdef PRINT_MATRIX
    // debug print W
    cout << "W" << endl << blockMatrixToMatrix(W.toDense()) << endl;
    cout << endl;
#endif

    // Init ea and eb random
    for (int i = 0; i < n; ++i) ea(i) = ARes::Random();
    for (int i = 0; i < m; ++i) eb(i) = BRes::Random();


#if 1
    // ======================== Dense Simple Solution (only for checking the correctness) ========================
    {
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
    }
#endif
}

int main(int argc, char* argv[])
{
    printVectorInstructions();

    simpleSchurTest();
    baBlockSchurTest();
    return 0;
}
