/**
 * This file is part of the Eigen Recursive Matrix Extension (ERME).
 *
 * Copyright (c) 2019 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "MatrixScalar.h"


namespace Eigen::Recursive
{
template <typename T>
struct RecursiveRandom
{
};


template <>
struct RecursiveRandom<double>
{
    static void set(double& value) { value = ((double)rand() / double(RAND_MAX)) * 2.0 - 1.0; }
};

template <>
struct RecursiveRandom<float>
{
    static void set(double& value) { value = ((float)rand() / float(RAND_MAX)) * 2.0f - 1.0f; }
};



template <typename G>
struct RecursiveRandom<MatrixScalar<G>>
{
    static void set(MatrixScalar<G>& value) { RecursiveRandom<G>::set(value.data); }
};


template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct RecursiveRandom<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    using Scalar         = _Scalar;
    using ChildExpansion = RecursiveRandom<_Scalar>;
    using MatrixType     = Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;

    static void set(MatrixType& A)
    {
        for (int i = 0; i < A.rows(); ++i)
        {
            for (int j = 0; j < A.cols(); ++j)
            {
                ChildExpansion::set(A(i, j));
            }
        }
    }
};


template <typename _Scalar, int _Options>
struct RecursiveRandom<Eigen::SparseMatrix<_Scalar, _Options>>
{
    using Scalar         = _Scalar;
    using ChildExpansion = RecursiveRandom<_Scalar>;
    using MatrixType     = Eigen::SparseMatrix<_Scalar, _Options>;


    static void set(MatrixType& A)
    {
        // Create a dense SparseMatrix
        std::vector<Triplet<Scalar>> triplets;
        for (int i = 0; i < A.rows(); ++i)
        {
            for (int j = 0; j < A.cols(); ++j)
            {
                Scalar s;
                ChildExpansion::set(s);

                triplets.emplace_back(i, j, s);
            }
        }
        A.setZero();
        A.setFromTriplets(triplets.begin(), triplets.end());
    }
};


template <typename T>
void setRandom(T& m)
{
    RecursiveRandom<T>::set(m);
}

}  // namespace Eigen::Recursive
