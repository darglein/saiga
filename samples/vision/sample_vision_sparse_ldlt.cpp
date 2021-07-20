/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#define CHOLMOD_OMP_NUM_THREADS 1
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 128

#include "saiga/core/math/Eigen_Compile_Checker.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/Thread/omp.h"
#include "saiga/core/util/table.h"
#include "saiga/vision/recursive/Recursive.h"
#include "saiga/vision/util/Random.h"

#include "Eigen/CholmodSupport"
#include "Eigen/SparseCholesky"

#include <fstream>
#include <set>
using namespace Saiga;
using namespace Eigen::Recursive;


static std::ofstream strm;

//#define SPARSITY_TEST
#define USE_BLOCKS
//#define LDLT_DEBUG


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
    using Block        = typename std::conditional<block_size == 1, double,
                                            Eigen::Matrix<double, block_size, block_size, Eigen::ColMajor>>::type;
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
    const double targetNNZ          = factor * 40 * 1000 * sqrt(block_size);
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

    const int nnzr = 1;
    //    const int n    = 9;
    const int n = factor * factor;
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

        //        initRandom(trips);
        initPoisson2D(trips);


        setRandom(b);
        //        for (int i = 0; i < n; ++i)
        //        {
        //            b(i) = RecursiveRandom<Vector>::get();
        //        }

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

        bx = expand(x);
        be = expand(b);

        cholmod_start(&m_cholmod);
        cholmod_defaults(&m_cholmod);
        m_cholmod.final_ll      = 0;
        m_cholmod.SPQR_nthreads = 1;
        m_cholmod.nmethods      = 1;
        computeOrdering();

        //        Random::setRandom(x);
        //        Random::setRandom(b);

#ifdef LDLT_DEBUG
        std::cout << expand(A) << std::endl << std::endl;
#endif
        //        std::cout << expand(Anoblock) << std::endl << std::endl;
        //        exit(0);
        //        std::cout << b.transpose() << std::endl;

        std::cout << "." << std::endl;
        std::cout << "Sparse LDLT test" << std::endl;
        std::cout << "Blocksize: " << block_size << std::endl;
        std::cout << "N: " << n << std::endl;
        std::cout << "Non zeros (per row): " << nnzr << std::endl;
        std::cout << "Non zeros: " << Anoblock.nonZeros() << std::endl;
        std::cout << "Density: " << density() << std::endl;
        std::cout << "." << std::endl;
        std::cout << std::endl;
    }


    ~Sparse_LDLT_TEST()
    {
        // cleanup cholmod types
        cholmod_finish(&m_cholmod);
        cholmod_free_work(&m_cholmod);
    }
    void addRank1Diagonal(std::vector<Eigen::Triplet<Block>>& trips)
    {
        for (int i = 0; i < n; ++i)
        {
            Vector dv;
            setRandom(dv);
            dv *= 100;
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

            setRandom(D);
            //            D = RecursiveRandom<Block>::get();
            // Make sure the matrix is positiv
            auto n = MultiplicativeNeutral<Block>::get();
            D += n * 500.0;
            //            D += 5;
            //            D.diagonal() += Vector::Ones() * 5;
            trips.emplace_back(i, i, D);
        }
    }

    void initRandom(std::vector<Eigen::Triplet<Block>>& trips)
    {
        trips.reserve(nnzr * n * 2);
        addRank1Diagonal(trips);

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
                    //                    std::cout << "insert " << i << " " << s << std::endl;
                    indices.insert(s);
                    ++k;
                }
            }
#else
            // Full random pattern
            auto indices = Random::uniqueIndices(nnzr, n);
            std::sort(indices.begin(), indices.end());
#endif

            // Set to 0 if you want a diagonal block matrix
#if 1
            for (auto j : indices)
            {
                if (i < j)
                {
                    Block b = RecursiveRandom<Block>::get();
                    trips.emplace_back(i, j, b);
                    trips.emplace_back(j, i, transpose(b));
                }
            }
#endif
        }
    }

    void initPoisson2D(std::vector<Eigen::Triplet<Block>>& trips)
    {
        trips.reserve(n);
        addRank1Diagonal(trips);

        int sideLength = sqrt(n);
        SAIGA_ASSERT(sideLength * sideLength == n);

        auto idx = [&](int i, int j) {
            //            while (i < 0) i += sideLength;
            //            while (j < 0) j += sideLength;
            //            while (i >= sideLength) i -= sideLength;
            //            while (j >= sideLength) j -= sideLength;
            return i * sideLength + j;
        };

        for (int i = 0; i < sideLength; ++i)
        {
            for (int j = 0; j < sideLength; ++j)
            {
                int c = idx(i, j);

                int dis                   = 2;
                std::array<int, 4> neighs = {idx(i, j + dis), idx(i, j - dis), idx(i, j + dis * 5),
                                             idx(i, j - dis * 5)};
                //                std::array<int, 4> neighs = {idx(i, j + dis), idx(i, j - dis), idx(i - dis, j), idx(i
                //                + dis, j)}; std::array<int, 8> neighs = {idx(i, j + 1),     idx(i, j - 1),     idx(i -
                //                1, j),
                //                                             idx(i + 1, j),     idx(i + 1, j + 1), idx(i - 1, j - 1),
                //                                             idx(i - 1, j + 1), idx(i + 1, j - 1)};

                for (auto ne : neighs)
                {
                    // boundary: do nothing
                    if (ne < 0 || ne >= n) continue;

                    if (c < ne)
                    {
                        //                        Block b = RecursiveRandom<Block>::get() * 1;
                        Block b;
                        setRandom(b);
                        trips.emplace_back(c, ne, b);
                        trips.emplace_back(ne, c, transpose(b));
                    }
                }
            }
        }
    }

    auto solveEigenDenseLDLT()
    {
        bx.setZero();
        auto Ae    = expand(A);
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
        bx.setZero();
        Eigen::SimplicialLDLT<decltype(Anoblock)> ldlt;

        float time = 0;
        {
            Saiga::ScopedTimer<float> timer(time);
            ldlt.compute(Anoblock);
            bx = ldlt.solve(be);
        }

#ifdef LDLT_DEBUG
        Eigen::MatrixXd L(ldlt.matrixL().eval());
        L.diagonal().setOnes();
        std::cout << "x: " << bx.transpose() << std::endl;
        std::cout << "L" << std::endl << L << std::endl << std::endl;
//        std::cout << "D" << std::endl << ldlt.vectorD().transpose() << std::endl << std::endl;
#endif

        double error = (Anoblock * bx - be).squaredNorm();
        return std::make_tuple(time, error, SAIGA_SHORT_FUNCTION);
    }
    auto solveEigenSparseLDLTRecursiveFlat()
    {
        bx.setZero();
        using MType = decltype(Anoblock);
        Eigen::RecursiveSimplicialLDLT<MType, Eigen::Lower> ldlt;
        ldlt.m_Pinv = permFull;
        ldlt.m_P    = permFull.inverse();
        float time  = 0;
        {
            Saiga::ScopedTimer<float> timer(time);
            ldlt.analyzePattern(Anoblock);
            ldlt.factorize(Anoblock);
        }
        bx           = ldlt.solve(be);
        double error = (Anoblock * bx - be).squaredNorm();
        return std::make_tuple(time, error, SAIGA_SHORT_FUNCTION);
    }



    auto solveEigenRecursiveSparseLDLT()
    {
        x.setZero();
        using LDLT = Eigen::RecursiveSimplicialLDLT<AType, Eigen::Lower>;
        LDLT ldlt;
        ldlt.m_Pinv = permBlock;
        ldlt.m_P    = permBlock.inverse();
        float time  = 0;
        {
            Saiga::ScopedTimer<float> timer(time);
            ldlt.analyzePattern(A);
            ldlt.factorize(A);
        }
        x = ldlt.solve(b);

#ifdef LDLT_DEBUG
        AType LA = ldlt.matrixL();
        Eigen::MatrixXd L(expand(LA));
        //        L.diagonal().setOnes();

        auto d = ldlt.vectorD();


        std::cout << "L" << std::endl << L << std::endl << std::endl;
        std::cout << "D" << std::endl << expand(d) << std::endl << std::endl;
#endif

        //        std::cout << expand(x).transpose() << std::endl;
        double error = expand((A * x - b).eval()).squaredNorm();
        return std::make_tuple(time, error, SAIGA_SHORT_FUNCTION);
    }


    auto solveEigenRecursiveSparseLDLT3()
    {
        x.setZero();
        using LDLT = Eigen::RecursiveSimplicialLDLT2<AType, Eigen::Lower>;
        LDLT ldlt;
        ldlt.m_Pinv = permBlock;
        ldlt.m_P    = permBlock.inverse();
        float time  = 0;
        {
            Saiga::ScopedTimer<float> timer(time);
            ldlt.analyzePattern(A);
            ldlt.factorize(A);
        }
        x = ldlt.solve(b);

#ifdef LDLT_DEBUG
        AType LA = ldlt.matrixL();
        Eigen::MatrixXd L(expand(LA));
        //        L.diagonal().setOnes();

        auto d = ldlt.vectorD();


//        std::cout << "L" << std::endl << L << std::endl << std::endl;
//        std::cout << "diagL" << std::endl << expand(ldlt.m_diagL) << std::endl << std::endl;
//        std::cout << "D" << std::endl << expand(d) << std::endl << std::endl;
//        std::cout << "Dinv" << std::endl << expand(ldlt.m_diag_inv) << std::endl << std::endl;
#endif

        //        std::cout << expand(x).transpose() << std::endl;
        double error = expand((A * x - b).eval()).squaredNorm();
        return std::make_tuple(time, error, SAIGA_SHORT_FUNCTION);
    }


    Eigen::PermutationMatrix<-1> permFull, permBlock;
    std::vector<int> orderingFull;
    std::vector<int> orderingBlock;

    void computeOrdering()
    {
        cholmod_sparse cholmod_matrix;
        cholmod_sparse cholmod_matrix_block;
        //        cholmod_common m_cholmod;

        const Eigen::SparseMatrix<double>& testasdf = Anoblock;
        cholmod_matrix                              = Eigen::viewAsCholmod(testasdf.selfadjointView<Eigen::Upper>());


        // copy structure to double sparse matrix
        Eigen::SparseMatrix<double> testBlock(n, n);
        testBlock.reserve(A.nonZeros());
        for (int i = 0; i < n + 1; ++i) testBlock.outerIndexPtr()[i] = A.outerIndexPtr()[i];
        for (int i = 0; i < A.nonZeros(); ++i) testBlock.innerIndexPtr()[i] = A.innerIndexPtr()[i];


        const auto& testBlockasdf = testBlock;
        cholmod_matrix_block      = Eigen::viewAsCholmod(testBlockasdf.template selfadjointView<Eigen::Upper>());

        m_cholmod.supernodal         = CHOLMOD_SIMPLICIAL;
        m_cholmod.nmethods           = 1;
        m_cholmod.method[0].ordering = CHOLMOD_AMD;
        m_cholmod.postorder          = false;


        orderingFull.resize(Anoblock.rows());
        orderingBlock.resize(A.rows());

        cholmod_amd(&cholmod_matrix, 0, 0, orderingFull.data(), &m_cholmod);
        cholmod_amd(&cholmod_matrix_block, 0, 0, orderingBlock.data(), &m_cholmod);


        // eigen types

        permFull.resize(Anoblock.rows());
        for (int i = 0; i < Anoblock.rows(); ++i)
        {
            permFull.indices()[i] = orderingFull[i];
        }

        permBlock.resize(A.rows());
        for (int i = 0; i < A.rows(); ++i)
        {
            permBlock.indices()[i] = orderingBlock[i];
        }
    }

    auto solveCholmodSimplicial()
    {
        auto t         = solveCholmod(false);
        std::get<2>(t) = SAIGA_SHORT_FUNCTION;
        return t;
    }

    auto solveCholmodSupernodal()
    {
        auto t         = solveCholmod(true);
        std::get<2>(t) = SAIGA_SHORT_FUNCTION;
        return t;
    }

    auto solveCholmod(bool supernodal)
    {
        bx.setZero();

        cholmod_factor* m_cholmodFactor = nullptr;
        cholmod_sparse cholmod_matrix;

        // Init
        const Eigen::SparseMatrix<double>& testasdf = Anoblock;
        cholmod_matrix                              = Eigen::viewAsCholmod(testasdf.selfadjointView<Eigen::Upper>());

        if (supernodal)
        {
            m_cholmod.supernodal = CHOLMOD_SUPERNODAL;
        }
        else
        {
            m_cholmod.supernodal = CHOLMOD_SIMPLICIAL;
        }

        m_cholmod.method[0].ordering = CHOLMOD_GIVEN;
        m_cholmod.postorder          = false;



        float time = 0;
        {
            Saiga::ScopedTimer<float> timer(time);
            m_cholmodFactor = cholmod_analyze_p(&cholmod_matrix, orderingFull.data(), 0, 0, &m_cholmod);
            cholmod_factorize(&cholmod_matrix, m_cholmodFactor, &m_cholmod);
        }


        cholmod_dense b_cd  = viewAsCholmod(be);
        cholmod_dense* x_cd = cholmod_solve(CHOLMOD_A, m_cholmodFactor, &b_cd, &m_cholmod);
        bx = Eigen::Matrix<double, -1, 1>::Map(reinterpret_cast<double*>(x_cd->x), bx.rows(), bx.cols());
        cholmod_free_factor(&m_cholmodFactor, &m_cholmod);
        cholmod_free_dense(&x_cd, &m_cholmod);

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

        //        std::cout << expand(x).transpose() << std::endl;
        double error = expand((A2 * x - b).eval()).squaredNorm();
        return std::make_tuple(time, error, SAIGA_SHORT_FUNCTION);
    }

    void solveSparseRecursive()
    {
        x.setZero();
        {
            SAIGA_BLOCK_TIMER();
            SparseRecursiveLDLT<AType2, VType> ldlt;
            ldlt.compute(A2);
            x = ldlt.solve(b);
        }

        //        std::cout << expand(x).transpose() << std::endl;
        std::cout << "Error: " << expand((A * x - b).eval()).squaredNorm() << std::endl << std::endl;

        //        std::cout << "L" << std::endl << expand(ldlt.L) << std::endl << std::endl;
        //        std::cout << "D" << std::endl << expand(ldlt.D.toDenseMatrix()) << std::endl << std::endl;
    }
    cholmod_common m_cholmod;
    Eigen::SparseMatrix<double> Anoblock;
    Eigen::Matrix<double, -1, 1> bx, be;
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

    for (int i = 0; i < 23; ++i)
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
    //    make_test(test, table, &LDLT::solveEigenSparseLDLTRecursiveFlat);
    //        make_test(test, table, &LDLT::solveEigenRecursiveSparseLDLTRowMajor);
    make_test(test, table, &LDLT::solveEigenRecursiveSparseLDLT);
    make_test(test, table, &LDLT::solveEigenRecursiveSparseLDLT3);

    //    make_test(test, table, &LDLT::solveEigenRecursiveSparseLDLT);

    make_test(test, table, &LDLT::solveCholmodSimplicial);
    make_test(test, table, &LDLT::solveCholmodSupernodal);
    //    make_test(test, table, &LDLT::solveEigenRecursiveCholmod);



    strm << std::endl;
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
            "eigen_recursive2,"
            "cholmod_simp,"
            "cholmod_super"
         << std::endl;

    OMP::setNumThreads(1);


    //    {
    //        LauncherLoop<2, 32 + 1, 1, 1, 32> l;
    //        l();
    //    }
    {
        //        LauncherLoop<8, 8 + 1, 1, 1, 32> l;
        //        l();
    }
    //    {
    //        LauncherLoop<16, 16 + 1, 1, 1, 32> l;
    //        l();
    //    }
    //    {
    //        LauncherLoop<24, 24 + 1, 1, 1, 32> l;
    //        l();
    //    }
}


void result_test()
{
    using LDLT = Sparse_LDLT_TEST<7, 1>;
    LDLT test;
    //    auto res = test.solveEigenSparseLDLT();
    //    std::cout << "Error: " << std::get<1>(res) << std::endl;
    //    res = test.solveEigenRecursiveCholmod();
    //    std::cout << "Error: " << std::get<1>(res) << std::endl;
    //    auto res = test.solveEigenRecursiveSparseLDLT();
    //    std::cout << "Error: " << std::get<1>(res) << std::endl;
    //    auto res = test.solveEigenRecursiveSparseLDLT();
    //    std::cout << "Error: " << std::get<1>(res) << std::endl;
    //    res = test.solveEigenRecursiveSparseLDLT();
    //    std::cout << "Error: " << std::get<1>(res) << std::endl;
    auto res = test.solveEigenRecursiveSparseLDLT3();
    std::cout << "Error: " << std::get<1>(res) << std::endl;

    //    test.solveEigenRecursiveSparseLDLTRowMajor();

#if 0

    using MT = Eigen::Matrix<double, 4, 4>;
    MT m;
    m.setRandom();
    m.triangularView<Eigen::Lower>() = m.triangularView<Eigen::Upper>().transpose();

    std::cout << m << std::endl << std::endl;

    Eigen::LDLT<MT> ldlt;


    Eigen::Matrix<double, 4, 1> b, x;
    b.setRandom();

    ldlt.compute(m);
    x = ldlt.solve(b);



    MT L = ldlt.matrixL();
    MT D;
    D.setZero();
    D.diagonal() = ldlt.vectorD();

    std::cout << L << std::endl << std::endl;
    std::cout << D << std::endl << std::endl;
    MT res = L * D * L.transpose();

    std::cout << res << std::endl << std::endl;

    auto error = ((m * x) - b).norm();
    std::cout << "error: " << error << std::endl;
#endif
}


int main(int, char**)
{
    Saiga::EigenHelper::checkEigenCompabitilty<2765>();
    Random::setSeed(15235);



    //    result_test();
    perf_test();

    std::cout << "Done." << std::endl;

    return 0;
}
