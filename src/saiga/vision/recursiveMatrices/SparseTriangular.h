/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/recursiveMatrices/RecursiveMatrices.h"

namespace Eigen
{
namespace internal
{
// forward substitution, col-major
template <typename T, typename Rhs, int Mode>
struct sparse_solve_triangular_selector<const Eigen::SparseMatrix<Saiga::MatrixScalar<T>>, Rhs, Mode, Lower, ColMajor>
{
    //    using Lhs = Eigen::SparseMatrix<Saiga::MatrixScalar<Eigen::Matrix<double, 2, 2>>>;
    using Lhs = Eigen::SparseMatrix<Saiga::MatrixScalar<T>>;
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
                        cout << "not unit :O" << endl;
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
template <typename T, typename Rhs, int Mode>
struct sparse_solve_triangular_selector<const Eigen::Transpose<const Eigen::SparseMatrix<Saiga::MatrixScalar<T>>>, Rhs,
                                        Mode, Upper, RowMajor>
{
    using Lhs = const Eigen::Transpose<const Eigen::SparseMatrix<Saiga::MatrixScalar<T>>>;

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
                    cout << "not unit :O" << endl;
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
                {
                    other.coeffRef(i, col) = tmp;
                }
                else
                {
                    cout << "not unit :O" << endl;
                    other.coeffRef(i, col).get() = Saiga::inverseCholesky(l_ii.get()) * tmp.get();
                }
                //                    other.coeffRef(i, col) = tmp / l_ii;
            }
        }
    }
};


template <int _SrcMode, int _DstMode, typename MatrixType, int DstOrder>
void permute_symm_to_symm_recursive(
    const MatrixType& mat,
    SparseMatrix<typename MatrixType::Scalar, DstOrder, typename MatrixType::StorageIndex>& _dest,
    const typename MatrixType::StorageIndex* perm)
{
    typedef typename MatrixType::StorageIndex StorageIndex;
    typedef typename MatrixType::Scalar Scalar;
    SparseMatrix<Scalar, DstOrder, StorageIndex>& dest(_dest.derived());
    typedef Matrix<StorageIndex, Dynamic, 1> VectorI;
    typedef evaluator<MatrixType> MatEval;
    typedef typename evaluator<MatrixType>::InnerIterator MatIterator;

    enum
    {
        SrcOrder          = MatrixType::IsRowMajor ? RowMajor : ColMajor,
        StorageOrderMatch = int(SrcOrder) == int(DstOrder),
        DstMode           = DstOrder == RowMajor ? (_DstMode == Upper ? Lower : Upper) : _DstMode,
        SrcMode           = SrcOrder == RowMajor ? (_SrcMode == Upper ? Lower : Upper) : _SrcMode
    };

    MatEval matEval(mat);

    Index size = mat.rows();
    VectorI count(size);
    count.setZero();
    dest.resize(size, size);
    for (StorageIndex j = 0; j < size; ++j)
    {
        StorageIndex jp = perm ? perm[j] : j;
        for (MatIterator it(matEval, j); it; ++it)
        {
            StorageIndex i = it.index();
            if ((int(SrcMode) == int(Lower) && i < j) || (int(SrcMode) == int(Upper) && i > j)) continue;

            StorageIndex ip = perm ? perm[i] : i;
            count[int(DstMode) == int(Lower) ? (std::min)(ip, jp) : (std::max)(ip, jp)]++;
        }
    }
    dest.outerIndexPtr()[0] = 0;
    for (Index j = 0; j < size; ++j) dest.outerIndexPtr()[j + 1] = dest.outerIndexPtr()[j] + count[j];
    dest.resizeNonZeros(dest.outerIndexPtr()[size]);
    for (Index j = 0; j < size; ++j) count[j] = dest.outerIndexPtr()[j];

    for (StorageIndex j = 0; j < size; ++j)
    {
        for (MatIterator it(matEval, j); it; ++it)
        {
            StorageIndex i = it.index();
            if ((int(SrcMode) == int(Lower) && i < j) || (int(SrcMode) == int(Upper) && i > j)) continue;

            StorageIndex jp = perm ? perm[j] : j;
            StorageIndex ip = perm ? perm[i] : i;

            Index k                 = count[int(DstMode) == int(Lower) ? (std::min)(ip, jp) : (std::max)(ip, jp)]++;
            dest.innerIndexPtr()[k] = int(DstMode) == int(Lower) ? (std::max)(ip, jp) : (std::min)(ip, jp);

            if (!StorageOrderMatch) std::swap(ip, jp);
            if (((int(DstMode) == int(Lower) && ip < jp) || (int(DstMode) == int(Upper) && ip > jp)))
                dest.valuePtr()[k].get() = Saiga::transpose(it.value().get());
            else
                dest.valuePtr()[k] = it.value();
        }
    }
}

#if 0
template <typename DstXprType, typename T, int Mode, typename Scalar>
struct Assignment<DstXprType, SparseSymmetricPermutationProduct<Eigen::SparseMatrix<Saiga::MatrixScalar<T>>, Mode>,
                  internal::assign_op<Scalar, typename Eigen::SparseMatrix<Saiga::MatrixScalar<T>>::Scalar>,
                  Sparse2Sparse>
{
    using MatrixType = Eigen::SparseMatrix<Saiga::MatrixScalar<T>>;
    typedef SparseSymmetricPermutationProduct<MatrixType, Mode> SrcXprType;
    typedef typename DstXprType::StorageIndex DstIndex;
    template <int Options>
    static void run(SparseMatrix<Scalar, Options, DstIndex>& dst, const SrcXprType& src,
                    const internal::assign_op<Scalar, typename MatrixType::Scalar>&)
    {
        // This is the same as eigens, because a full permuation doesn't need to propagate transposes
        SparseMatrix<Scalar, (Options & RowMajor) == RowMajor ? ColMajor : RowMajor, DstIndex> tmp;
        internal::permute_symm_to_fullsymm<Mode>(src.matrix(), tmp, src.perm().indices().data());
        dst = tmp;
    }

    template <typename DestType, unsigned int DestMode>
    static void run(SparseSelfAdjointView<DestType, DestMode>& dst, const SrcXprType& src,
                    const internal::assign_op<Scalar, typename MatrixType::Scalar>&)
    {
        cout << "permute col" << endl;
        internal::permute_symm_to_symm_recursive<Mode, DestMode>(src.matrix(), dst.matrix(),
                                                                 src.perm().indices().data());
    }
};
#endif


template <typename DstXprType, typename T, int _Options, int Mode, typename Scalar>
struct Assignment<
    DstXprType, SparseSymmetricPermutationProduct<Eigen::SparseMatrix<Saiga::MatrixScalar<T>, _Options>, Mode>,
    internal::assign_op<Scalar, typename Eigen::SparseMatrix<Saiga::MatrixScalar<T>, _Options>::Scalar>, Sparse2Sparse>
{
    using MatrixType = Eigen::SparseMatrix<Saiga::MatrixScalar<T>, _Options>;
    typedef SparseSymmetricPermutationProduct<MatrixType, Mode> SrcXprType;
    typedef typename DstXprType::StorageIndex DstIndex;
    template <int Options>
    static void run(SparseMatrix<Scalar, Options, DstIndex>& dst, const SrcXprType& src,
                    const internal::assign_op<Scalar, typename MatrixType::Scalar>&)
    {
        // This is the same as eigens, because a full permuation doesn't need to propagate transposes
        SparseMatrix<Scalar, (Options & RowMajor) == RowMajor ? ColMajor : RowMajor, DstIndex> tmp;
        internal::permute_symm_to_fullsymm<Mode>(src.matrix(), tmp, src.perm().indices().data());
        dst = tmp;
    }

    template <typename DestType, unsigned int DestMode>
    static void run(SparseSelfAdjointView<DestType, DestMode>& dst, const SrcXprType& src,
                    const internal::assign_op<Scalar, typename MatrixType::Scalar>&)
    {
        //        cout << "permute row" << endl;
        internal::permute_symm_to_symm_recursive<Mode, DestMode>(src.matrix(), dst.matrix(),
                                                                 src.perm().indices().data());
    }
};


}  // namespace internal
}  // namespace Eigen
