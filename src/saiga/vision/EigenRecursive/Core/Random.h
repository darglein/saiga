/**
 * Copyright (c) 2017 Darius Rückert
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
    static double get() { return ((double)rand() / RAND_MAX) * 2.0 - 1.0; }
};

template <>
struct RecursiveRandom<float>
{
    static float get() { return ((float)rand() / RAND_MAX) * 2.0f - 1.0f; }
};



template <typename G>
struct RecursiveRandom<MatrixScalar<G>>
{
    static MatrixScalar<G> get() { return makeMatrixScalar(RecursiveRandom<G>::get()); }
};


template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct RecursiveRandom<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    using Scalar         = _Scalar;
    using ChildExpansion = RecursiveRandom<_Scalar>;
    using MatrixType     = Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;

    static MatrixType get()
    {
        MatrixType A;

        for (int i = 0; i < A.rows(); ++i)
        {
            for (int j = 0; j < A.cols(); ++j)
            {
                A(i, j) = ChildExpansion::get();
            }
        }
        return A;
    }
};

}  // namespace Eigen::Recursive
