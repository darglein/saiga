/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/random.h"
#include "saiga/vision/Random.h"
#include "saiga/vision/recursiveMatrices/RecursiveMatrices.h"
#include "saiga/vision/recursiveMatrices/SparseCholesky.h"

#include "Eigen/CholmodSupport"
#include "Eigen/SparseCholesky"
#include "SimplicialCholesky.h"
namespace Eigen
{
namespace internal
{
// forward substitution, col-major
// template <typename T, typename Rhs, int Mode>
// struct sparse_solve_triangular_selector<Eigen::SparseMatrix<Saiga::MatrixScalar<T>, Eigen::ColMajor>, Rhs, Mode,
// Lower,
//                                        ColMajor>
//{
//    using Lhs = Eigen::SparseMatrix<Saiga::MatrixScalar<T>, Eigen::ColMajor>;

template <typename Rhs, int Mode>
struct sparse_solve_triangular_selector<const Eigen::SparseMatrix<Saiga::MatrixScalar<Eigen::Matrix<double, 2, 2>>>,
                                        Rhs, Mode, Lower, ColMajor>
{
    using Lhs = Eigen::SparseMatrix<Saiga::MatrixScalar<Eigen::Matrix<double, 2, 2>>>;
    typedef typename Lhs::Scalar LScalar;
    typedef typename Rhs::Scalar RScalar;
    typedef evaluator<Lhs> LhsEval;
    typedef typename evaluator<Lhs>::InnerIterator LhsIterator;
    static void run(const Lhs& lhs, Rhs& other)
    {
        LhsEval lhsEval(lhs);
        for (Index col = 0; col < other.cols(); ++col)
        {
            for (Index i = 0; i < lhs.cols(); ++i)
            {
                RScalar& tmp = other.coeffRef(i, col);
                //                if (tmp != Scalar(0))  // optimization when other is actually sparse
                {
                    LhsIterator it(lhsEval, i);
                    while (it && it.index() < i) ++it;
                    if (!(Mode & UnitDiag))
                    {
                        eigen_assert(it && it.index() == i);
                        tmp.get() = Saiga::inverseCholesky(it.value().get()) * tmp.get();
                    }
                    if (it && it.index() == i) ++it;
                    for (; it; ++it) other.coeffRef(it.index(), col).get() -= it.value().get() * tmp.get();
                }
            }
        }
    }
};



// backward substitution, row-major
template <typename Rhs, int Mode>
struct sparse_solve_triangular_selector<
    const Eigen::Transpose<const Eigen::SparseMatrix<Saiga::MatrixScalar<Eigen::Matrix<double, 2, 2>>>>, Rhs, Mode,
    Upper, RowMajor>
{
    using Lhs = const Eigen::Transpose<const Eigen::SparseMatrix<Saiga::MatrixScalar<Eigen::Matrix<double, 2, 2>>>>;

    typedef typename Lhs::Scalar LScalar;
    typedef typename Rhs::Scalar RScalar;

    typedef evaluator<Lhs> LhsEval;
    typedef typename evaluator<Lhs>::InnerIterator LhsIterator;
    static void run(const Lhs& lhs, Rhs& other)
    {
        LhsEval lhsEval(lhs);
        for (Index col = 0; col < other.cols(); ++col)
        {
            for (Index i = lhs.rows() - 1; i >= 0; --i)
            {
                RScalar tmp = other.coeff(i, col);
                LScalar l_ii(0);
                LhsIterator it(lhsEval, i);
                while (it && it.index() < i) ++it;
                if (!(Mode & UnitDiag))
                {
                    eigen_assert(it && it.index() == i);
                    l_ii = it.value();
                    ++it;
                }
                else if (it && it.index() == i)
                    ++it;
                for (; it; ++it)
                {
                    tmp.get() -= it.value().get().transpose() * other.coeff(it.index(), col).get();
                }

                if (Mode & UnitDiag)
                    other.coeffRef(i, col) = tmp;
                else
                    other.coeffRef(i, col).get() = Saiga::inverseCholesky(l_ii.get()) * tmp.get();
                //                    other.coeffRef(i, col) = tmp / l_ii;
            }
        }
    }
};


}  // namespace internal
}  // namespace Eigen

using namespace Saiga;

#define USE_BLOCKS

class Sparse_LDLT_TEST
{
   public:
#ifdef USE_BLOCKS
    static constexpr int block_size = 2;
    using Block                     = Eigen::Matrix<double, block_size, block_size>;
    using Vector                    = Eigen::Matrix<double, block_size, 1>;
    using AType                     = Eigen::SparseMatrix<MatrixScalar<Block>>;
    using AType2                    = Eigen::SparseMatrix<MatrixScalar<Block>, Eigen::RowMajor>;
    using VType                     = Eigen::Matrix<MatrixScalar<Vector>, -1, 1>;
#else
    const int block_size = 1;
    using Block          = double;
    using Vector         = double;
    using AType          = Eigen::SparseMatrix<Block>;
    using AType2         = Eigen::SparseMatrix<Block, Eigen::RowMajor>;
    using VType          = Eigen::Matrix<Vector, -1, 1>;
#endif


    Sparse_LDLT_TEST(int n)
    {
        int nnzr = 10;
        A.resize(n, n);
        A2.resize(n, n);
        Anoblock.resize(n * block_size, n * block_size);
        x.resize(n);
        b.resize(n);

        std::vector<Eigen::Triplet<Block>> trips;
        trips.reserve(nnzr * n * 2);

        for (int i = 0; i < n; ++i)
        {
            auto indices = Random::uniqueIndices(nnzr, n);
            std::sort(indices.begin(), indices.end());

            for (auto j : indices)
            {
                if (i != j)
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
            Block D = dv * dv.transpose();
#else
            Block D = dv * transpose(dv);
#endif

            // Make sure the matrix is positiv
            auto n = MultiplicativeNeutral<Block>::get();
            D += n * 5.0;
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
            for (int i = 0; i < block_size; ++i)
                for (int j = 0; j < block_size; ++j)
                {
                    trips_no_block.emplace_back(t.row() * block_size + i, t.col() * block_size + j, t.value()(i, j));
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
    }

    void solveDenseLDLT()
    {
        if (A.rows() > 100) return;
        auto Ae = expand(A);
        auto be = expand(b);
        auto bx = expand(x);

        {
            SAIGA_BLOCK_TIMER();
            bx = Ae.ldlt().solve(be);
        }
        cout << "Error: " << (Ae * bx - be).squaredNorm() << endl << endl;
    }

    void solveSparseLDLT()
    {
        auto be = expand(b);
        auto bx = expand(x);
        {
            SAIGA_BLOCK_TIMER();
            Eigen::SimplicialLDLT<decltype(Anoblock)> ldlt;
            ldlt.compute(Anoblock);
            bx = ldlt.solve(be);
        }

        cout << "Error: " << (Anoblock * bx - be).squaredNorm() << endl << endl;
    }

    void solveCholmod()
    {
        auto be = expand(b);
        auto bx = expand(x);
        {
            SAIGA_BLOCK_TIMER();
            Eigen::CholmodSimplicialLDLT<decltype(Anoblock)> ldlt;
            //            Eigen::CholmodSupernodalLLT<decltype(Anoblock)> ldlt;
            ldlt.compute(Anoblock);
            bx = ldlt.solve(be);
        }

        cout << "Error: " << (Anoblock * bx - be).squaredNorm() << endl << endl;
    }

    void solveSparseLDLTRecursive()
    {
        x.setZero();

        {
            SAIGA_BLOCK_TIMER();
            using LDLT = Eigen::SimplicialLDLT2<AType, Eigen::Lower>;
            LDLT ldlt;
            ldlt.compute(A);
            x = ldlt.solve(b);
        }

        //        cout << expand(x).transpose() << endl;
        cout << "Error: " << expand((A * x - b).eval()).squaredNorm() << endl << endl;


#if 0
        Eigen::Matrix<double, -1, -1> L          = expand(ldlt.matrixL().toDense());
        L.diagonal()                             = Eigen::Matrix<double, -1, 1>::Ones(L.rows());
        Eigen::Matrix<double, -1, -1> D          = expand(ldlt.vectorD().asDiagonal().toDenseMatrix());
        Eigen::Matrix<double, -1, -1> P_block    = ldlt.permutationP().toDenseMatrix().cast<double>();
        Eigen::Matrix<double, -1, -1> Pinv_block = ldlt.permutationPinv().toDenseMatrix().cast<double>();


        int block_size = 2;

        Eigen::Matrix<double, -1, -1> P(P_block.rows() * block_size, P_block.cols() * block_size);
        P.setZero();
        for (int i = 0; i < P_block.rows(); ++i)
        {
            for (int j = 0; j < P_block.cols(); ++j)
            {
                if (P_block(i, j) == 1)
                {
                    P.block<2, 2>(i * block_size, j * block_size) = Block::Identity();
                }
            }
        }


        Eigen::Matrix<double, -1, -1> Pinv = P.transpose();

#    if 0
        cout << "L2" << endl << L << endl;
        cout << "D2" << endl << D << endl;
        cout << "L*D*LT'" << endl << (L * D * L.transpose()).eval() << endl;
        cout << "P*L*D*LT*P'" << endl << (Pinv * L * D * L.transpose() * P).eval() << endl;
#    endif
        cout << "P" << endl << P << endl;


        cout << "L" << endl << L << endl;

        auto per = ldlt.permutationP();


        //        A.selfadjointView<Eigen::Upper>().twistedBy(per);

#    if 0
        for (int i = 0; i < per.rows(); ++i)
        {
            for (int j = 0; j < per.cols(); ++j)
            {
                int k = per.indices()(i);
                int n = per.indices()(j);

                bool src = i < j;
                bool dst = k < n;

                if (src == dst)
                {
                    L.block<2, 2>(i * block_size, j * block_size) = L.block<2, 2>(i * block_size, j * block_size);
                }
                else
                {
                    L.block<2, 2>(i * block_size, j * block_size) =
                        L.block<2, 2>(i * block_size, j * block_size).transpose();
                }
            }
        }
#    endif

        cout << "L2" << endl << L << endl;


        //        ldlt.permutationP()

        auto m_ldlt = (L * D * L.transpose()).eval();

        auto m_pldltp = Pinv * m_ldlt * P;
        //        m_pldltp.setZero();

#    if 0
        auto per = ldlt.permutationP();

        for (int i = 0; i < per.rows(); ++i)
        {
            for (int j = 0; j < per.cols(); ++j)
            {
                int k = per.indices()(i);
                int n = per.indices()(j);

                bool src = i < j;
                bool dst = k < n;

                if (src == dst)
                {
                    m_pldltp.block<2, 2>(i * block_size, j * block_size) =
                        m_ldlt.block<2, 2>(k * block_size, n * block_size);
                }
                else
                {
                    m_pldltp.block<2, 2>(i * block_size, j * block_size) =
                        m_ldlt.block<2, 2>(k * block_size, n * block_size).transpose();
                }
            }
        }
#    endif
        //        auto m_pldltp = (Pinv * L * D * L.transpose() * P).eval();

        cout << "A" << endl << expand(A) << endl;
        cout << "aldlt" << endl << m_ldlt << endl;
        cout << "aldlt" << endl << m_pldltp << endl;
        cout << endl;
        cout << per.indices().transpose() << endl;

        double AError = (expand(A) - m_pldltp).norm();
        cout << "A error: " << AError << endl;

#endif
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


int main(int, char**)
{
    Saiga::EigenHelper::checkEigenCompabitilty<2765>();

    Random::setSeed(15235);

    Sparse_LDLT_TEST test(1000);
    test.solveDenseLDLT();
    test.solveSparseLDLT();
    test.solveCholmod();
    test.solveSparseLDLTRecursive();
    //    test.solveSparseRecursive();
    cout << "Done." << endl;
    return 0;
}
