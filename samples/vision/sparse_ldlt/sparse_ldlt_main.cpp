/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#define CHOLMOD_OMP_NUM_THREADS 1
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 128


#include "saiga/core/util/random.h"
#include "saiga/core/util/table.h"
#include "saiga/vision/Random.h"
#include "saiga/vision/recursiveMatrices/RecursiveMatrices.h"
#include "saiga/vision/recursiveMatrices/RecursiveSimplicialCholesky.h"
#include "saiga/vision/recursiveMatrices/SparseCholesky.h"

#include "Eigen/CholmodSupport"
#include "Eigen/SparseCholesky"

#include <fstream>
#include <set>
using namespace Saiga;



static std::ofstream strm;

//#define SPARSITY_TEST
#define USE_BLOCKS
#define LDLT_DEBUG


template <int block_size2, int factor>
class Sparse_LDLT_TEST
{
   public:
#ifdef USE_BLOCKS

#    ifdef SPARSITY_TEST
    // use fixed block size
    static constexpr int block_size = 4;
#    else
    static constexpr int block_size = block_size2;
#    endif
    using Block =
        typename std::conditional<block_size == 1, double, Eigen::Matrix<double, block_size, block_size>>::type;
    using Vector       = typename std::conditional<block_size == 1, double, Eigen::Matrix<double, block_size, 1>>::type;
    using WrappedBlock = typename std::conditional<block_size == 1, double, MatrixScalar<Block>>::type;
    using WrappedVector = typename std::conditional<block_size == 1, double, MatrixScalar<Vector>>::type;
    using AType         = Eigen::SparseMatrix<WrappedBlock>;
    using AType2        = Eigen::SparseMatrix<WrappedBlock, Eigen::RowMajor>;
    using VType         = Eigen::Matrix<WrappedVector, -1, 1>;
#else
    static constexpr int block_size = 1;
    using Block                     = double;
    using Vector                    = double;
    using AType                     = Eigen::SparseMatrix<Block>;
    using AType2                    = Eigen::SparseMatrix<Block, Eigen::RowMajor>;
    using VType                     = Eigen::Matrix<Vector, -1, 1>;
#endif



#ifdef SPARSITY_TEST

    const int n    = 100 * 1000 / (int)log(block_size2);
    const int nnzr = 1 * block_size2 / (int)log(block_size2);
#else

#    if 0
    const double targetNNZ          = factor * 400 * 1000 * sqrt(block_size);
    const double targetDensity      = 0.001;

    const double nnzBlocks = targetNNZ / double(block_size * block_size);
    const double zBlocks   = 1.0 / targetDensity * nnzBlocks;

    const int n    = sqrt(zBlocks);
    const int nnzr = std::max<int>(nnzBlocks / n, 1);
#    else

    // const double targetNNZ = 1000 * 1000 * 4;
    //    const int nnzr = 4;
    //    const int n    = 4000;
    // const double nnzBlocks = targetNNZ / double(block_size * block_size);
    // const int n            = nnzBlocks / nnzr;

    const int nnzr = 4;
    const int n    = 4;
#    endif
#endif

    double density() { return (double)Anoblock.nonZeros() / double(double(n) * n * block_size * block_size); }

    Sparse_LDLT_TEST()
    {
        A.resize(n, n);
        A2.resize(n, n);
        Anoblock.resize(n * block_size, n * block_size);
        x.resize(n);
        b.resize(n);

        std::vector<Eigen::Triplet<Block>> trips;
        trips.reserve(nnzr * n * 2);

        for (int i = 0; i < n; ++i)
        {
#if 0
            // "Diagonal like" pattern
            std::set<int> indices;
            for (int k = 0; k < nnzr;)
            {
                int s = Random::gaussRand(i, nnzr * 0.5);
                while (s < 0)
                {
                    s += n;
                }
                while (s >= n)
                {
                    s -= n;
                }
                if (s != i && indices.count(s) == 0)
                {
                    //                    cout << "insert " << i << " " << s << endl;
                    indices.insert(s);
                    ++k;
                }
            }
#else
            // Full random pattern
            auto indices = Random::uniqueIndices(nnzr, n);
            std::sort(indices.begin(), indices.end());
#endif

            for (auto j : indices)
            {
                if (i < j)
                {
                    Block b = RecursiveRandom<Block>::get();
                    trips.emplace_back(i, j, b);
                    trips.emplace_back(j, i, transpose(b));
                }
            }

            // Make sure we have a symmetric diagonal block
            //            Vector dv = Random::sampleDouble(-1, 1);
            Vector dv = RecursiveRandom<Vector>::get();


#ifdef USE_BLOCKS
            Block D;
            if constexpr (block_size == 1)
                D = dv * dv;
            else
                D = dv * dv.transpose();
//            Block D = dv * dv.transpose();
#else
            Block D = dv * transpose(dv);
#endif

            // Make sure the matrix is positiv
            auto n = MultiplicativeNeutral<Block>::get();
            D += n * 500.0;
            //            D += 5;
            //            D.diagonal() += Vector::Ones() * 5;
            trips.emplace_back(i, i, D);

            b(i) = RecursiveRandom<Vector>::get();
        }
        A.setFromTriplets(trips.begin(), trips.end());
        A.makeCompressed();

        A2.setFromTriplets(trips.begin(), trips.end());
        A2.makeCompressed();


        std::vector<Eigen::Triplet<double>> trips_no_block;
        trips_no_block.reserve(nnzr * n * 2 * block_size * block_size);
        for (auto t : trips)
        {
            if constexpr (block_size == 1)
            {
                trips_no_block.emplace_back(t.row(), t.col(), t.value());
            }
            else
            {
                for (int i = 0; i < block_size; ++i)
                {
                    for (int j = 0; j < block_size; ++j)
                    {
                        trips_no_block.emplace_back(t.row() * block_size + i, t.col() * block_size + j,
                                                    t.value()(i, j));
                    }
                }
            }
        }
        Anoblock.setFromTriplets(trips_no_block.begin(), trips_no_block.end());
        Anoblock.makeCompressed();



        //        Random::setRandom(x);
        //        Random::setRandom(b);

        //        cout << expand(A) << endl << endl;
        //        cout << expand(Anoblock) << endl << endl;
        //        exit(0);
        //        cout << b.transpose() << endl;

        cout << "." << endl;
        cout << "Sparse LDLT test" << endl;
        cout << "Blocksize: " << block_size << endl;
        cout << "N: " << n << endl;
        cout << "Non zeros (per row): " << nnzr << endl;
        cout << "Non zeros: " << Anoblock.nonZeros() << endl;
        cout << "Density: " << density() << endl;
        cout << "." << endl;
        cout << endl;
    }

    auto solveEigenDenseLDLT()
    {
        x.setZero();
        auto Ae    = expand(A);
        auto be    = expand(b);
        auto bx    = expand(x);
        float time = 0;
        {
            Saiga::ScopedTimer<float> timer(time);
            bx = Ae.ldlt().solve(be);
        }
        double error = (Ae * bx - be).squaredNorm();
        return std::make_tuple(time, error, SAIGA_SHORT_FUNCTION);
    }

    auto solveEigenSparseLDLT()
    {
        x.setZero();
        auto be = expand(b);
        auto bx = expand(x);
        Eigen::SimplicialLDLT<decltype(Anoblock)> ldlt;

        float time = 0;
        {
            Saiga::ScopedTimer<float> timer(time);
            ldlt.compute(Anoblock);
            bx = ldlt.solve(be);
        }

#ifdef LDLT_DEBUG
        Eigen::MatrixXd L(ldlt.matrixL());
        L.diagonal().setOnes();
        cout << "L" << endl << L << endl << endl;
        cout << "D" << endl << ldlt.vectorD().transpose() << endl << endl;
#endif

        double error = (Anoblock * bx - be).squaredNorm();
        return std::make_tuple(time, error, SAIGA_SHORT_FUNCTION);
    }

    auto solveCholmodSimplicial()
    {
        x.setZero();
        auto be = expand(b);
        auto bx = expand(x);
        Eigen::CholmodSimplicialLDLT<decltype(Anoblock)> ldlt;
        //        Eigen::CholmodSimplicialLLT<decltype(Anoblock)> ldlt;

        float time = 0;
        {
            Saiga::ScopedTimer<float> timer(time);
            ldlt.compute(Anoblock);
            bx = ldlt.solve(be);
        }
        double error = (Anoblock * bx - be).squaredNorm();
        return std::make_tuple(time, error, SAIGA_SHORT_FUNCTION);
    }

    auto solveCholmodSupernodal()
    {
        x.setZero();
        auto be = expand(b);
        auto bx = expand(x);
        //        Eigen::CholmodSupernodalLLT<decltype(Anoblock)> ldlt;
        Eigen::CholmodDecomposition<decltype(Anoblock)> ldlt;

        ldlt.cholmod().supernodal    = CHOLMOD_SUPERNODAL;
        ldlt.cholmod().final_ll      = 0;
        ldlt.cholmod().SPQR_nthreads = 1;
        //        ldlt.cholmod().final_asis = 1;

        float time = 0;
        {
            Saiga::ScopedTimer<float> timer(time);
            ldlt.compute(Anoblock);
            bx = ldlt.solve(be);
        }
        double error = (Anoblock * bx - be).squaredNorm();
        return std::make_tuple(time, error, SAIGA_SHORT_FUNCTION);
    }



    auto solveEigenRecursiveSparseLDLT()
    {
        x.setZero();
        using LDLT = Eigen::RecursiveSimplicialLDLT<AType, Eigen::Lower>;

        LDLT ldlt;
        float time = 0;
        {
            Saiga::ScopedTimer<float> timer(time);
            ldlt.compute(A);
            //            ldlt.analyzePattern(A);
            //            ldlt.factorize(A);
            x = ldlt.solve(b);
        }

#ifdef LDLT_DEBUG
        AType LA = ldlt.matrixL();
        Eigen::MatrixXd L(expand(LA));
        L.diagonal().setOnes();

        auto d = ldlt.vectorD();


        cout << "L" << endl << L << endl << endl;
        cout << "D" << endl << expand(d) << endl << endl;
#endif

        //        cout << expand(x).transpose() << endl;
        double error = expand((A * x - b).eval()).squaredNorm();
        return std::make_tuple(time, error, SAIGA_SHORT_FUNCTION);
    }

    auto solveEigenRecursiveCholmod()
    {
        // Convert recursive to flat matrix

        Eigen::SparseMatrix<double> test;
        using AType = decltype(test);



        x.setZero();

        auto be = expand(b);
        auto bx = expand(x);
        //        Eigen::CholmodSupernodalLLT<decltype(Anoblock)> ldlt;
        Eigen::CholmodDecomposition<decltype(test)> ldlt;

        ldlt.cholmod().supernodal    = CHOLMOD_SUPERNODAL;
        ldlt.cholmod().final_ll      = 0;
        ldlt.cholmod().SPQR_nthreads = 1;
        //        ldlt.cholmod().final_asis = 1;

        float time = 0;
        {
            typedef typename AType::StorageIndex StorageIndex;
            Saiga::ScopedTimer<float> timer(time);
            sparseBlockToFlatMatrix(A, test);

            test = Anoblock;

            //            cout << expand(A) << endl << endl;
            //            cout << expand(test) << endl << endl;

            const auto& testasdf = test;
            auto cholmod_matrix  = Eigen::viewAsCholmod(testasdf.selfadjointView<Eigen::Upper>());

            double m_shiftOffset[2]         = {0, 0};
            cholmod_factor* m_cholmodFactor = nullptr;
            cholmod_common m_cholmod        = ldlt.cholmod();

            m_cholmodFactor = Eigen::internal::cm_analyze<StorageIndex>(cholmod_matrix, m_cholmod);
            SAIGA_ASSERT(m_cholmodFactor);

            Eigen::internal::cm_factorize_p<StorageIndex>(&cholmod_matrix, m_shiftOffset, 0, 0, m_cholmodFactor,
                                                          m_cholmod);
            SAIGA_ASSERT(m_cholmodFactor->minor == m_cholmodFactor->n);



            cholmod_dense b_cd  = viewAsCholmod(be);
            cholmod_dense* x_cd = Eigen::internal::cm_solve<StorageIndex>(CHOLMOD_A, *m_cholmodFactor, b_cd, m_cholmod);

            bx = Eigen::Matrix<double, -1, 1>::Map(reinterpret_cast<double*>(x_cd->x), bx.rows(), bx.cols());

            Eigen::internal::cm_free_factor<StorageIndex>(m_cholmodFactor, m_cholmod);
            Eigen::internal::cm_free_dense<StorageIndex>(x_cd, m_cholmod);
        }
        double error = (Anoblock * bx - be).squaredNorm();
        return std::make_tuple(time, error, SAIGA_SHORT_FUNCTION);
    }


    auto solveEigenRecursiveSparseLDLTRowMajor()
    {
        // Convert recursive to flat matrix

        //        Eigen::SparseMatrix<double> test;
        //        sparseBlockToFlatMatrix(A2, test);

        x.setZero();
        using LDLT = Eigen::RecursiveSimplicialLDLT<AType2, Eigen::Upper>;

        float time = 0;
        {
            Saiga::ScopedTimer<float> timer(time);
            LDLT ldlt;
            ldlt.compute(A2);
            //            ldlt.analyzePattern(A);
            //            ldlt.factorize(A);
            x = ldlt.solve(b);
        }

        //        cout << expand(x).transpose() << endl;
        double error = expand((A2 * x - b).eval()).squaredNorm();
        return std::make_tuple(time, error, SAIGA_SHORT_FUNCTION);
    }

    void solveSparseRecursive()
    {
        x.setZero();
        {
            SAIGA_BLOCK_TIMER();
            Saiga::SparseRecursiveLDLT<AType2, VType> ldlt;
            ldlt.compute(A2);
            x = ldlt.solve(b);
        }

        //        cout << expand(x).transpose() << endl;
        cout << "Error: " << expand((A * x - b).eval()).squaredNorm() << endl << endl;

        //        cout << "L" << endl << expand(ldlt.L) << endl << endl;
        //        cout << "D" << endl << expand(ldlt.D.toDenseMatrix()) << endl << endl;
    }
    Eigen::SparseMatrix<double> Anoblock;
    AType A;
    AType2 A2;
    VType x, b;
};

template <typename LDLT, typename T>
float make_test(LDLT& ldlt, Saiga::Table& tab, T f)
{
    std::vector<double> time;
    std::string name;
    float error;

    for (int i = 0; i < 51; ++i)
    {
        auto [time2, error2, name2] = (ldlt.*f)();
        time.push_back(time2);
        name  = name2;
        error = error2;
    }

    Saiga::Statistics s(time);
    float t = s.median;
    auto t2 = t / ldlt.Anoblock.nonZeros() * 1000;
    tab << name << t << t2 << error;

    strm << "," << t;
    return t;
}

template <int block_size, int factor>
void run()
{
    using LDLT = Sparse_LDLT_TEST<block_size, factor>;
    LDLT test;
    strm << test.n << "," << test.Anoblock.nonZeros() << "," << block_size << "," << test.density();

    Saiga::Table table{{35, 15, 15, 15}};
    table.setFloatPrecision(6);
    table << "Solver"
          << "Time (ms)"
          << "Time/NNZ (us)"
          << "Error";
    //    if (test.A.rows() < 100) make_test(test, table, &LDLT::solveEigenDenseLDLT);
    //    make_test(test, table, &LDLT::solveEigenSparseLDLT);
    //    make_test(test, table, &LDLT::solveEigenRecursiveSparseLDLTRowMajor);
    //    make_test(test, table, &LDLT::solveEigenRecursiveSparseLDLT);

    //    make_test(test, table, &LDLT::solveEigenRecursiveSparseLDLT);

    //    make_test(test, table, &LDLT::solveCholmodSimplicial);
    make_test(test, table, &LDLT::solveCholmodSupernodal);
    make_test(test, table, &LDLT::solveEigenRecursiveCholmod);



    strm << endl;
}



template <int START, int END, int ADD, int MULT, int factor>
struct LauncherLoop
{
    void operator()()
    {
        {
            run<START, factor>();
        }
        LauncherLoop<START * MULT + ADD, END, ADD, MULT, factor> l;
        l();
    }
};

template <int END, int ADD, int MULT, int factor>
struct LauncherLoop<END, END, ADD, MULT, factor>
{
    void operator()() {}
};


void perf_test()
{
    strm.open("sparse_ldlt_benchmark.csv");
    strm << "n,nnz,block_size,density,"
            "eigen_recursive,"
            "cholmod_simp,"
            "cholmod_super"
         << endl;
    //        LauncherLoop<2, 8 + 1, 16> l;
    //    LauncherLoop<1, 32 + 1, 8> l;

#ifdef SPARSITY_TEST
    {
        LauncherLoop<1, 32 * 2, 0, 2, 2> l;
        l();
    }

#else
    {
        omp_set_num_threads(1);
        LauncherLoop<2, 2 + 1, 1, 1, 2> l;

        //        LauncherLoop<4, 4 + 1, 1, 1, 16> l;
        l();
    }
    {
        //        LauncherLoop<16, 32 + 2, 2, 1, 4> l;
        //        l();
    }
//    {
//        LauncherLoop<8, 8 + 1, 2> l;
//        l();
//    }
//    {
//        LauncherLoop<16, 16 + 1, 2> l;
//        l();
//    }
//    {
//        LauncherLoop<32, 32 + 1, 2> l;
//        l();
//    }
#endif
}


void result_test()
{
    using LDLT = Sparse_LDLT_TEST<2, 1>;
    LDLT test;
    test.solveEigenSparseLDLT();

    auto res = test.solveEigenRecursiveSparseLDLT();

    cout << "Error: " << std::get<1>(res) << endl;
    //    test.solveEigenRecursiveSparseLDLTRowMajor();
}

int main(int, char**)
{
    Saiga::EigenHelper::checkEigenCompabitilty<2765>();
    Random::setSeed(15235);



    result_test();
    //    perf_test();

    cout << "Done." << endl;

    return 0;
}
