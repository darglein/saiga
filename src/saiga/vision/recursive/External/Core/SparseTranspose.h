/**
 * This file contains (modified) code from the Eigen library.
 * Eigen License:
 *
 * Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
 * Copyright (C) 2007-2011 Benoit Jacob <jacob.benoit.1@gmail.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla
 * Public License v. 2.0. If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * ======================
 *
 * The modifications are part of the Eigen Recursive Matrix Extension (ERME).
 * ERME License:
 *
 * Copyright (c) 2019 Darius Rückert
 * Licensed under the MIT License.
 */

#pragma once
#include "SparseHelper.h"
#include "Transpose.h"

#include <iostream>

namespace Eigen::Recursive
{
/**
 * Sparse Matrix Transposition.
 * This is basically a copy and paste from Eigen/src/SparseCore/SparseMatrix.h :: operator=
 *
 * The only difference is that we call transpose recursivly on each element when assigning them.
 *
 * There are also two additional methods that only transpose the structure/values.
 * This is used for optimization problems with well known structures. In these cases
 * the structure can be precomputed.
 *
 */

template <typename G, typename H, int options>
void transpose(const Eigen::SparseMatrix<G, options>& other, Eigen::SparseMatrix<H, options>& dest)
{
    static_assert(options == Eigen::RowMajor, "todo");
    using SparseMatrix = Eigen::SparseMatrix<G, Eigen::RowMajor>;

    using namespace Eigen;
    //        SparseMatrix dest(other.rows(),other.cols());
    //    dest.resize(other.rows(), other.cols());
    dest.resize(other.cols(), other.rows());
    Eigen::Map<typename SparseMatrix::IndexVector>(dest.outerIndexPtr(), dest.outerSize()).setZero();

    // pass 1
    // FIXME the above copy could be merged with that pass
    for (Index j = 0; j < other.outerSize(); ++j)
        for (typename SparseMatrix::InnerIterator it(other, j); it; ++it) ++dest.outerIndexPtr()[it.index()];

    // prefix sum
    Index count = 0;
    typename SparseMatrix::IndexVector positions(dest.outerSize());
    for (Index j = 0; j < dest.outerSize(); ++j)
    {
        auto tmp                = dest.outerIndexPtr()[j];
        dest.outerIndexPtr()[j] = count;
        positions[j]            = count;
        count += tmp;
    }
    dest.outerIndexPtr()[dest.outerSize()] = count;
    // alloc
    //        dest.m_data.resize(count);
    dest.reserve(count);
    // pass 2
    for (Index j = 0; j < other.outerSize(); ++j)
    {
        for (typename SparseMatrix::InnerIterator it(other, j); it; ++it)
        {
            Index pos                  = positions[it.index()]++;
            dest.innerIndexPtr()[pos]  = j;
            dest.valuePtr()[pos].get() = transpose(it.value()).get();
        }
    }
}


template <typename G, typename H, int options>
void transposeStructureOnly(const Eigen::SparseMatrix<G, options>& other, Eigen::SparseMatrix<H, options>& dest)
{
    static_assert(options == Eigen::RowMajor, "todo");
    using SparseMatrix = Eigen::SparseMatrix<G, Eigen::RowMajor>;



    using namespace Eigen;
    //        SparseMatrix dest(other.rows(),other.cols());
    dest.resize(other.cols(), other.rows());
    Eigen::Map<typename SparseMatrix::IndexVector>(dest.outerIndexPtr(), dest.outerSize()).setZero();

    // pass 1
    // FIXME the above copy could be merged with that pass
    for (Index j = 0; j < other.outerSize(); ++j)
        for (typename SparseMatrix::InnerIterator it(other, j); it; ++it) ++dest.outerIndexPtr()[it.index()];

    // prefix sum
    Index count = 0;
    typename SparseMatrix::IndexVector positions(dest.outerSize());
    for (Index j = 0; j < dest.outerSize(); ++j)
    {
        auto tmp                = dest.outerIndexPtr()[j];
        dest.outerIndexPtr()[j] = count;
        positions[j]            = count;
        count += tmp;
    }
    dest.outerIndexPtr()[dest.outerSize()] = count;
    // alloc
    dest.reserve(count);
    // pass 2
    for (Index j = 0; j < other.outerSize(); ++j)
    {
        //        int op = other.outerIndexPtr()[j];
        int i = 0;
        for (typename SparseMatrix::InnerIterator it(other, j); it; ++it, ++i)
        {
            Index pos                 = positions[it.index()]++;
            dest.innerIndexPtr()[pos] = j;
        }
    }
}

template <typename G, typename H, int options>
void transposeStructureOnly_omp(const Eigen::SparseMatrix<G, options>& other, Eigen::SparseMatrix<H, options>& dest,
                                std::vector<int>& transposeTargets)
{
    static_assert(options == Eigen::RowMajor, "todo");
    using SparseMatrix = Eigen::SparseMatrix<G, Eigen::RowMajor>;



    using namespace Eigen;
    //        SparseMatrix dest(other.rows(),other.cols());
    dest.resize(other.cols(), other.rows());
    Eigen::Map<typename SparseMatrix::IndexVector>(dest.outerIndexPtr(), dest.outerSize()).setZero();

    // pass 1
    // FIXME the above copy could be merged with that pass
    for (Index j = 0; j < other.outerSize(); ++j)
        for (typename SparseMatrix::InnerIterator it(other, j); it; ++it) ++dest.outerIndexPtr()[it.index()];

    // prefix sum
    Index count = 0;
    typename SparseMatrix::IndexVector positions(dest.outerSize());
    for (Index j = 0; j < dest.outerSize(); ++j)
    {
        auto tmp                = dest.outerIndexPtr()[j];
        dest.outerIndexPtr()[j] = count;
        positions[j]            = count;
        count += tmp;
    }
    dest.outerIndexPtr()[dest.outerSize()] = count;
    // alloc
    dest.reserve(count);
    transposeTargets.resize(count);
    // pass 2
    for (Index j = 0; j < other.outerSize(); ++j)
    {
        int op = other.outerIndexPtr()[j];
        int i  = 0;
        for (typename SparseMatrix::InnerIterator it(other, j); it; ++it, ++i)
        {
            int rel                   = op + i;
            Index pos                 = positions[it.index()]++;
            transposeTargets[rel]     = pos;
            dest.innerIndexPtr()[pos] = j;
        }
    }
}


template <typename G, typename H, int options>
void transposeValueOnly(const Eigen::SparseMatrix<G, options>& other, Eigen::SparseMatrix<H, options>& dest)
{
    static_assert(options == Eigen::RowMajor, "todo");
    using SparseMatrix = Eigen::SparseMatrix<G, Eigen::RowMajor>;
    using namespace Eigen;

    std::vector<int> positions(dest.outerSize(), 0);

    for (Index j = 0; j < other.outerSize(); ++j)
    {
        for (typename SparseMatrix::InnerIterator it(other, j); it; ++it)
        {
            Index pos = dest.outerIndexPtr()[it.index()] + positions[it.index()]++;

            dest.valuePtr()[pos].get() = transpose(it.value()).get();
        }
    }
}


template <typename G, typename H, int options>
void transposeValueOnly_omp(const Eigen::SparseMatrix<G, options>& other, Eigen::SparseMatrix<H, options>& dest,
                            const std::vector<int>& transposeTargets)
{
    static_assert(options == Eigen::RowMajor, "todo");
    using SparseMatrix = Eigen::SparseMatrix<G, Eigen::RowMajor>;
    using namespace Eigen;

    //    std::vector<int> positions(dest.outerSize(), 0);

#pragma omp for
    for (Index j = 0; j < other.outerSize(); ++j)
    {
        int op = other.outerIndexPtr()[j];
        int i  = 0;
        for (typename SparseMatrix::InnerIterator it(other, j); it; ++it, ++i)
        {
            int rel = op + i;
            int pos = transposeTargets[rel];
            //            Index pos = dest.outerIndexPtr()[it.index()] + positions[it.index()]++;

            dest.valuePtr()[pos].get() = transpose(it.value()).get();
        }
    }
}



}  // namespace Eigen::Recursive
