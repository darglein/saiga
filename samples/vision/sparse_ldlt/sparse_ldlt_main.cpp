/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/random.h"
#include "saiga/vision/Random.h"
#include "saiga/vision/recursiveMatrices/RecursiveMatrices.h"
#include "saiga/vision/recursiveMatrices/SparseCholesky.h"

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



class Sparse_LDLT_TEST
{
   public:
    //    using Block  = double;
    //    using Vector = double;

    using Block  = Eigen::Matrix<double, 2, 2>;
    using Vector = Eigen::Matrix<double, 2, 1>;

    using AType  = Eigen::SparseMatrix<MatrixScalar<Block>>;
    using AType2 = Eigen::SparseMatrix<MatrixScalar<Block>, Eigen::RowMajor>;
    using VType  = Eigen::Matrix<MatrixScalar<Vector>, -1, 1>;
    Sparse_LDLT_TEST(int n)
    {
        int nnzr = 1;
        A.resize(n, n);
        A2.resize(n, n);
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
                    Block b = Block::Random();
                    trips.emplace_back(i, j, b);
                    trips.emplace_back(j, i, b.transpose());
                }
            }

            // Make sure we have a symmetric diagonal block
            //            Vector dv = Random::sampleDouble(-1, 1);
            Vector dv = Vector::Random();
            Block D   = dv * dv.transpose();

            // Make sure the matrix is positiv
            //            D += 5;
            D.diagonal() += Vector::Ones() * 5;
            trips.emplace_back(i, i, D);

            b(i) = Vector::Random();
        }
        A.setFromTriplets(trips.begin(), trips.end());
        A.makeCompressed();

        A2.setFromTriplets(trips.begin(), trips.end());
        A2.makeCompressed();


        //        Random::setRandom(x);
        //        Random::setRandom(b);

        cout << expand(A) << endl << endl;
        //        cout << b.transpose() << endl;
    }

    void solveDenseLDLT()
    {
        auto Ae         = expand(A);
        auto be         = expand(b);
        decltype(be) xe = Ae.ldlt().solve(be);
        cout << xe.transpose() << endl;
        cout << "Dense LDLT error: " << (Ae * xe - be).squaredNorm() << endl;
    }

    void solveSparseLDLT()
    {
        //        Eigen::SimplicialLDLT<AType, Eigen::Upper> ldlt;
        //        ldlt.compute(A);
        //        x = ldlt.solve(b);
        //        cout << "Sparse LDLT error: " << (A * x - b).squaredNorm() << endl;
        //        cout << ldlt.permutationP().toDenseMatrix() << endl;
    }

    void solveSparseLDLT2()
    {
        x.setZero();
        using LDLT = Eigen::SimplicialLDLT2<AType, Eigen::Upper>;
        LDLT ldlt;

        //        int n = A.rows();

        //        LDLT::CholMatrixType tmp(n, n);
        //        LDLT::ConstCholMatrixPtr pmat;
        //        ldlt.ordering(A, pmat, tmp);
        //        ldlt.analyzePattern_preordered(*pmat, true);
        //        ldlt.factorize_preordered<true>(*pmat);
        ldlt.compute(A);
        x = ldlt.solve(b);

        cout << expand(x).transpose() << endl;
        cout << "Sparse LDLT error: " << expand((A * x - b).eval()).squaredNorm() << endl;
    }

    void solveSparseLDLT3()
    {
        x.setZero();
        Saiga::SparseRecursiveLDLT<AType2, VType> ldlt;
        ldlt.compute(A2);
        x = ldlt.solve(b);

        cout << expand(x).transpose() << endl;
        cout << "Sparse LDLT error: " << expand((A * x - b).eval()).squaredNorm() << endl;

        cout << expand(ldlt.L) << endl << endl;
        cout << expand(ldlt.D.toDenseMatrix()) << endl << endl;
    }
    AType A;
    AType2 A2;
    VType x, b;
};


int main(int argc, char* argv[])
{
    Saiga::EigenHelper::checkEigenCompabitilty<2765>();

    Random::setSeed(38670346483);

    Sparse_LDLT_TEST test(2);
    //    test.solveDenseLDLT();
    //    test.solveSparseLDLT();
    test.solveSparseLDLT3();
    test.solveSparseLDLT2();
    cout << "Done." << endl;
    return 0;
}
