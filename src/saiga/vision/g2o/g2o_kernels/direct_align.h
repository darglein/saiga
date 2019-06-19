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

#pragma once


#include "saiga/core/image/image.h"

#include "sophus_sba.h"

namespace g2o
{
using Saiga::SE3;
using Saiga::Vec2;
using Saiga::Vec3;



class EdgeSE3DirectAlign : public g2o::BaseUnaryEdge<1, Vec2, VertexSE3>
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3DirectAlign() {}

    bool read(std::istream& is) { return false; }
    bool write(std::ostream& os) const { return false; }

    void computeError()
    {
        const VertexSE3* v1 = static_cast<const VertexSE3*>(_vertices[0]);
        SE3 se3             = v1->estimate();

        Vec3 pc = se3 * worldPoint;

        auto x = pc(0);
        auto y = pc(1);
        auto z = pc(2);

        Vec2 ip(fx * x / z + cx, fy * y / z + cy);

        ip *= (scale);

        auto cI = I.inter(ip(1), ip(0));
        auto cD = D.inter(ip(1), ip(0));


        _error(0) = _measurement(0) - cI;
        //        _error(1) = bf/z - bf*cD;

        //        std::cout << _error.transpose() << std::endl;
        SAIGA_ASSERT(_error.allFinite());
    }


    virtual void linearizeOplus()
    {
        // TODO:
        // 1. Compute image gradients
        // 2. Compute the entries below!
        const VertexSE3* v1 = static_cast<const VertexSE3*>(_vertices[0]);
        SE3 se3             = v1->estimate();

        Vec3 pc = se3 * worldPoint;

        auto x  = pc(0);
        auto y  = pc(1);
        auto z  = pc(2);
        auto zz = z * z;

        Vec2 ip(fx * x / z + cx, fy * y / z + cy);

        ip *= (scale);

        auto gx = gIx.inter(ip(1), ip(0));
        auto gy = gIy.inter(ip(1), ip(0));


        //======================= res(0) ==================
        //        static_assert(_jacobianOplusXi.RowsAtCompileTime == 2, "Rows incorrect");
        //        static_assert(_jacobianOplusXi.ColsAtCompileTime == 6, "Cols incorrect");

        auto z_inv  = 1.0 / z;
        auto zz_inv = 1.0 / zz;

        // Translation
        _jacobianOplusXi(0, 0) = (gx * z_inv) + 0;
        _jacobianOplusXi(0, 1) = 0 + (gy * z_inv);
        _jacobianOplusXi(0, 2) = -(gx * x * zz_inv) - (gy * y * zz_inv);
        // Rotation
        _jacobianOplusXi(0, 3) = (gx * -y * x * zz_inv) - (gy * (1 + y * y * zz_inv));
        _jacobianOplusXi(0, 4) = (gx * (1 + x * x * zz_inv)) + (gy * x * y * zz_inv);
        _jacobianOplusXi(0, 5) = (gx * -y * z_inv) + (gy * x * z_inv);

        _jacobianOplusXi.row(0) *= -1.0;

#if 0
        //======================= res(0) ==================

        gx = gDx.inter(ip(1),ip(0));
        gy = gDy.inter(ip(1),ip(0));

        // Translation
        _jacobianOplusXi(1, 0) = -1.0/zz *  0 - ( (gx * z_inv)      + 0);
        _jacobianOplusXi(1, 1) = -1.0/zz *  0 - ( 0                 + (gy * z_inv));
        _jacobianOplusXi(1, 2) = -1.0/zz *  1 - (-(gx * x * zz_inv) - (gy * y * zz_inv));

        _jacobianOplusXi(1, 3) = -1.0/zz *  y - ((gx * -y*x*zz_inv)     - (gy * (1 + y*y * zz_inv)));
        _jacobianOplusXi(1, 4) = -1.0/zz * -x - ((gx * (1+x*x*zz_inv))  + (gy * x*y*zz_inv));
        _jacobianOplusXi(1, 5) = -1.0/zz *  0 - ((gx * -y*z_inv)        + (gy * x*z_inv));

        _jacobianOplusXi.row(1) *= bf;
#endif
        //        std::cout << _jacobianOplusXi << std::endl;
        //        std::cout << (1.0/zz*bf) << " " << bf << std::endl;
        //        exit(0);

        SAIGA_ASSERT(_jacobianOplusXi.allFinite());
    }

    const double scale = 0.5;

    Vec3 worldPoint;

    number_t fx, fy, cx, cy, bf;

    // Intensity and gradients in x, and y direction
    Saiga::ImageView<float> I, gIx, gIy;

    // Depth and gradients
    Saiga::ImageView<float> D, gDx, gDy;
};


}  // namespace g2o
