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

/*

NOTE: these functions have been adapted from the LDL library:

LDL Copyright (c) 2005 by Timothy A. Davis.  All Rights Reserved.

LDL License:

    Your use or distribution of LDL or any modified version of
    LDL implies that you agree to this License.

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301
    USA

    Permission is hereby granted to use or copy this program under the
    terms of the GNU LGPL, provided that the Copyright, this License,
    and the Availability of the original version is retained on all copies.
    User documentation of any code that uses this code or any modified
    version of this code must cite the Copyright, this License, the
    Availability note, and "Used by permission." Permission to modify
    the code and to distribute modified code is granted, provided the
    Copyright, this License, and the Availability note are retained,
    and a notice that the code was modified is included.
 */

//#ifndef LDTL_TEST_IMPL
//#define LDTL_TEST_IMPL
#pragma once

#include "../Core.h"
#include "Cholesky.h"
#include "Eigen/src/Core/util/NonMPL2.h"
#include "RecursiveSimplicialCholesky.h"


namespace Eigen
{
template <typename Derived>
void RecursiveSimplicialCholeskyBase<Derived>::analyzePattern_preordered(const CholMatrixType& ap, bool doLDLT)
{
    const StorageIndex size = StorageIndex(ap.rows());
    m_matrix.resize(size, size);
    m_parent.resize(size);
    m_nonZerosPerCol.resize(size);
    m_diag.resize(size);
    m_diag_inv.resize(size);

    rowCache.resize(size);
    pattern.resize(size);
    tags.resize(size);
    ei_declare_aligned_stack_constructed_variable(StorageIndex, tags, size, 0);

    for (StorageIndex k = 0; k < size; ++k)
    {
        /* L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
        m_parent[k]         = -1; /* parent of k is not yet known */
        tags[k]             = k;  /* mark node k as visited */
        m_nonZerosPerCol[k] = 0;  /* count of nonzeros in column k of L */
        for (typename CholMatrixType::InnerIterator it(ap, k); it; ++it)
        {
            StorageIndex i = it.index();
            if (i < k)
            {
                /* follow path from i to root of etree, stop at flagged node */
                for (; tags[i] != k; i = m_parent[i])
                {
                    /* find parent of i if not yet determined */
                    if (m_parent[i] == -1) m_parent[i] = k;
                    m_nonZerosPerCol[i]++; /* L (k,i) is nonzero */
                    tags[i] = k;           /* mark i as visited */
                }
            }
        }
    }

    /* construct Lp index array from m_nonZerosPerCol column counts */
    StorageIndex* Lp = m_matrix.outerIndexPtr();
    Lp[0]            = 0;
    for (StorageIndex k = 0; k < size; ++k) Lp[k + 1] = Lp[k] + m_nonZerosPerCol[k] + (doLDLT ? 0 : 1);

    m_matrix.resizeNonZeros(Lp[size]);

    m_isInitialized     = true;
    m_info              = Success;
    m_analysisIsOk      = true;
    m_factorizationIsOk = false;
}


template <typename Derived>
template <bool DoLDLT>
void RecursiveSimplicialCholeskyBase<Derived>::factorize_preordered(const CholMatrixType& ap)
{
    static_assert(DoLDLT == true, "only ldlt supported");
    using std::sqrt;

    eigen_assert(m_analysisIsOk && "You must first call analyzePattern()");
    eigen_assert(ap.rows() == ap.cols());
    eigen_assert(m_parent.size() == ap.rows());
    eigen_assert(m_nonZerosPerCol.size() == ap.rows());

    const StorageIndex size = StorageIndex(ap.rows());
    const StorageIndex* Lp  = m_matrix.outerIndexPtr();
    StorageIndex* Li        = m_matrix.innerIndexPtr();
    Scalar* Lx              = m_matrix.valuePtr();

#if 0
    // doesn't make a difference
    ei_declare_aligned_stack_constructed_variable(Scalar, rowCache, size, 0);
    ei_declare_aligned_stack_constructed_variable(StorageIndex, pattern, size, 0);
    ei_declare_aligned_stack_constructed_variable(StorageIndex, tags, size, 0);
#else

#endif

    bool ok = true;


    for (StorageIndex k = 0; k < size; ++k)
    {
        // compute nonzero pattern of kth row of L, in topological order


        // We could clear the complete cache here, but it's more efficient to clear
        // only k. (see line below)
        //        for (auto& rc : rowCache) rc = Saiga::AdditiveNeutral<Scalar>::get();
        rowCache[k] = Recursive::AdditiveNeutral<Scalar>::get();  // Y(0:k) is now all zero

        StorageIndex top    = size;  // stack for pattern is empty
        tags[k]             = k;     // mark node k as visited
        m_nonZerosPerCol[k] = 0;     // count of nonzeros in column k of L

        // The columns of ap stored in the upper triangular part.
        // This is the current row of the original matrix.
        for (typename CholMatrixType::InnerIterator it(ap, k); it; ++it)
        {
            StorageIndex i = it.index();
            //            std::cout << "idx: " << i << "<=" << k << std::endl;
            if (i <= k)
            {
                /* scatter A(i,k) into Y (sum duplicates) */
                // Note: we need a + here if the matrix contains duplicates
                //                rowCache[i] = transpose(it.value());
                rowCache[i].get() = it.value().get().transpose();
                Index len;
                for (len = 0; tags[i] != k; i = m_parent[i])
                {
                    pattern[len++] = i; /* L(k,i) is nonzero */
                    tags[i]        = k; /* mark i as visited */
                                        //                    if (i != it.index())
                    //                    std::cout << "found additional non zero L: " << k << "," << i << std::endl;
                }
                while (len > 0)
                {
                    pattern[--top] = pattern[--len];
                }

                //                std::cout << "stack " << top << " is now " << pattern[top] << " for i " << i << " " <<
                //                it.index() << std::endl;
            }
        }

        /* compute numerical values kth row of L (a sparse triangular solve) */
        // This is the diagonal element of the current row
        RealScalar diagElement = (rowCache[k]);
        rowCache[k]            = Recursive::AdditiveNeutral<Scalar>::get();

        for (; top < size; ++top)
        {
            Index i  = pattern[top]; /* pattern[top:n-1] is pattern of L(:,k) */
            Index p2 = Lp[i] + m_nonZerosPerCol[i];
            Index p  = Lp[i];


            //            std::cout << "Pattern " << k << "," << i << std::endl;
            Scalar target = rowCache[i]; /* get and clear Y(i) */
            rowCache[i]   = Recursive::AdditiveNeutral<Scalar>::get();

            auto& invDiagUp = m_diag_inv[i];

            Scalar prop_tmp;


            Scalar yi = target;
            prop_tmp  = yi;
            target    = yi * invDiagUp;



            // Propagate into everything to the right
            //            diagElement -= (prop_tmp)*transpose(target);
            diagElement.get() -= prop_tmp.get() * target.get().transpose();
            //            for (p = Lp[i]; p < p2; ++p)
            for (Index k = p; k < p2; ++k)
            {
                //                rowCache[Li[p]] -= prop_tmp * transpose(Lx[p]);
                rowCache[Li[k]].get() -= prop_tmp.get() * Lx[k].get().transpose();
            }

            Li[p2] = k;
            Lx[p2] = target;
            ++m_nonZerosPerCol[i]; /* increment count of nonzeros in col i */
        }

        m_diag[k] = diagElement;
        // Recursive call
#if 1
        m_diag_inv[k] = Recursive::inverseCholesky(m_diag[k]);
#else
        eldlt.compute(m_diag[k].get());
        m_diag_inv[k] = eldlt.solve(Scalar::M::Identity());
#endif
    }

    m_info              = ok ? Success : NumericalIssue;
    m_factorizationIsOk = true;
}

}  // namespace Eigen

//#endif
