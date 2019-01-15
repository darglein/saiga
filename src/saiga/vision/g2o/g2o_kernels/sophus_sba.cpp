// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "sophus_sba.h"

#include "g2o/core/factory.h"
#include "g2o/stuff/macros.h"

namespace g2o
{
using Saiga::SE3;
using Saiga::Vec2;
using Saiga::Vec3;

using namespace std;
using namespace Eigen;

VertexPoint::VertexPoint() : BaseVertex<3, Vector3>() {}

VertexSE3::VertexSE3() : BaseVertex<6, SE3>() {}



EdgeSE3PointProject::EdgeSE3PointProject() : BaseBinaryEdge<2, Vector2, VertexPoint, VertexSE3>() {}



void EdgeSE3PointProject::linearizeOplus()
{
    VertexSE3* vj = static_cast<VertexSE3*>(_vertices[1]);
    SE3 T(vj->estimate());
    VertexPoint* vi   = static_cast<VertexPoint*>(_vertices[0]);
    Vector3 xyz       = vi->estimate();
    Vector3 xyz_trans = T * (xyz);

    number_t x = xyz_trans[0];
    number_t y = xyz_trans[1];
    number_t z = xyz_trans[2];
    auto zz    = z * z;
    auto zinv  = 1 / z;
    auto zzinv = 1 / zz;
    //    number_t z_2 = z * z;

    Matrix3 R = T.so3().matrix();

    auto& fx = intr.fx;
    auto& fy = intr.fy;

    _jacobianOplusXi(0, 0) = R(0, 0) * zinv - x * R(2, 0) * zzinv;
    _jacobianOplusXi(0, 1) = R(0, 1) * zinv - x * R(2, 1) * zzinv;
    _jacobianOplusXi(0, 2) = R(0, 2) * zinv - x * R(2, 2) * zzinv;

    _jacobianOplusXi(1, 0) = R(1, 0) * zinv - y * R(2, 0) * zzinv;
    _jacobianOplusXi(1, 1) = R(1, 1) * zinv - y * R(2, 1) * zzinv;
    _jacobianOplusXi(1, 2) = R(1, 2) * zinv - y * R(2, 2) * zzinv;

    _jacobianOplusXi.row(0) *= -weight * fx;
    _jacobianOplusXi.row(1) *= -weight * fy;

    // Jacobian for the SE3 2x6

    // Translation
    _jacobianOplusXj(0, 0) = zinv;
    _jacobianOplusXj(0, 1) = 0;
    _jacobianOplusXj(0, 2) = -x * zzinv;
    _jacobianOplusXj(1, 0) = 0;
    _jacobianOplusXj(1, 1) = zinv;
    _jacobianOplusXj(1, 2) = -y * zzinv;


    // Rotation
    _jacobianOplusXj(0, 3) = -y * x * zzinv;
    _jacobianOplusXj(0, 4) = (1 + (x * x) * zzinv);
    _jacobianOplusXj(0, 5) = -y * zinv;
    _jacobianOplusXj(1, 3) = (-1 - (y * y) * zzinv);
    _jacobianOplusXj(1, 4) = x * y * zzinv;
    _jacobianOplusXj(1, 5) = x * zinv;

    _jacobianOplusXj.row(0) *= -weight * fx;
    _jacobianOplusXj.row(1) *= -weight * fy;
}



EdgeSE3PointProjectDepth::EdgeSE3PointProjectDepth() : BaseBinaryEdge<3, Vector3, VertexPoint, VertexSE3>() {}


void EdgeSE3PointProjectDepth::linearizeOplus()
{
    VertexPoint* vi = static_cast<VertexPoint*>(_vertices[0]);
    VertexSE3* vj   = static_cast<VertexSE3*>(_vertices[1]);
    Vector3 wp      = vi->estimate();
    SE3 se3(vj->estimate());

    Vector3 pc = se3 * wp;

    Matrix3 R = se3.so3().matrix();

    number_t x   = pc[0];
    number_t y   = pc[1];
    number_t z   = pc[2];
    number_t z_2 = z * z;


    auto& fx = intr.fx;
    auto& fy = intr.fy;

    _jacobianOplusXi(0, 0) = -fx * R(0, 0) / z + fx * x * R(2, 0) / z_2;
    _jacobianOplusXi(0, 1) = -fx * R(0, 1) / z + fx * x * R(2, 1) / z_2;
    _jacobianOplusXi(0, 2) = -fx * R(0, 2) / z + fx * x * R(2, 2) / z_2;

    _jacobianOplusXi(1, 0) = -fy * R(1, 0) / z + fy * y * R(2, 0) / z_2;
    _jacobianOplusXi(1, 1) = -fy * R(1, 1) / z + fy * y * R(2, 1) / z_2;
    _jacobianOplusXi(1, 2) = -fy * R(1, 2) / z + fy * y * R(2, 2) / z_2;

#if 0
  _jacobianOplusXi(2, 0) = bf * R(2, 0) / z_2;
  _jacobianOplusXi(2, 1) = bf * R(2, 1) / z_2;
  _jacobianOplusXi(2, 2) = bf * R(2, 2) / z_2;
#else
    _jacobianOplusXi(2, 0) = _jacobianOplusXi(0, 0) - bf * R(2, 0) / z_2;
    _jacobianOplusXi(2, 1) = _jacobianOplusXi(0, 1) - bf * R(2, 1) / z_2;
    _jacobianOplusXi(2, 2) = _jacobianOplusXi(0, 2) - bf * R(2, 2) / z_2;
#endif

    _jacobianOplusXi.row(0) *= weights(0);
    _jacobianOplusXi.row(1) *= weights(0);
    _jacobianOplusXi.row(2) *= weights(1);

    // translation
    _jacobianOplusXj(0, 0) = -1. / z * fx;
    _jacobianOplusXj(0, 1) = 0;
    _jacobianOplusXj(0, 2) = x / z_2 * fx;

    _jacobianOplusXj(1, 0) = 0;
    _jacobianOplusXj(1, 1) = -1. / z * fy;
    _jacobianOplusXj(1, 2) = y / z_2 * fy;

#if 0
  _jacobianOplusXj(2, 0) = 0;
  _jacobianOplusXj(2, 1) = 0;
  _jacobianOplusXj(2, 2) = bf / z_2;
#else
    _jacobianOplusXj(2, 0) = _jacobianOplusXj(0, 0) - 0;
    _jacobianOplusXj(2, 1) = _jacobianOplusXj(0, 1) - 0;
    _jacobianOplusXj(2, 2) = _jacobianOplusXj(0, 2) - bf / z_2;
#endif

    // Rotation
    _jacobianOplusXj(0, 3) = x * y / z_2 * fx;
    _jacobianOplusXj(0, 4) = -(1 + (x * x / z_2)) * fx;
    _jacobianOplusXj(0, 5) = y / z * fx;

    _jacobianOplusXj(1, 3) = (1 + y * y / z_2) * fy;
    _jacobianOplusXj(1, 4) = -x * y / z_2 * fy;
    _jacobianOplusXj(1, 5) = -x / z * fy;

#if 0
  _jacobianOplusXj(2, 3) = + bf * y / z_2;
  _jacobianOplusXj(2, 4) = - bf * x / z_2;
  _jacobianOplusXj(2, 5) = 0;
#else
    _jacobianOplusXj(2, 3) = _jacobianOplusXj(0, 3) - bf * y / z_2;
    _jacobianOplusXj(2, 4) = _jacobianOplusXj(0, 4) + bf * x / z_2;
    _jacobianOplusXj(2, 5) = _jacobianOplusXj(0, 5) - 0;
#endif

    _jacobianOplusXj.row(0) *= weights(0);
    _jacobianOplusXj.row(1) *= weights(0);
    _jacobianOplusXj.row(2) *= weights(1);
}

}  // namespace g2o
