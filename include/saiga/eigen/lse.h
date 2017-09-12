/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/eigen/eigen.h"
#include <Eigen/QR>
#include <Eigen/SVD>

namespace Saiga {

/**
 * Methods for computing the Nullspace of a general square singular matrix A (det(A)=0).
 *
 * The matrix must be rank deficient of degree 1, meaning that for example a 3x3 matrix must
 * have rank 2.
 *
 * The result vector is the non trivial solution (up to a scale) of
 * Ax=0
 */


template<typename matrix_t, typename vector_t>
void solveNullspaceLU(const matrix_t& A, vector_t& x){
    x = A.fullPivLu().kernel();
    x.normalize();
}

template<typename matrix_t, typename vector_t>
void solveNullspaceQR(const matrix_t& A, vector_t& x){
    //    auto qr = A.transpose().householderQr();
    auto qr = A.transpose().colPivHouseholderQr();
    matrix_t Q = qr.householderQ();
    x = Q.col(A.rows() - 1);
    x.normalize();
}

template<typename matrix_t, typename vector_t>
void solveNullspaceSVD(const matrix_t& A, vector_t& x){
//    SAIGA_ASSERT(A.rows() == A.cols());
    x = A.jacobiSvd(Eigen::ComputeFullV).matrixV().col( A.rows() - 1 );
    x.normalize();
}

}
