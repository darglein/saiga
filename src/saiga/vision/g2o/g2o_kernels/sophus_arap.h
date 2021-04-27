/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 *
 * This file is a partial copy of LSD-SLAM by Jakob Engel.
 * LSD-SLAM was resleased under the GNU General Public license.
 * For more information see <http://vision.in.tum.de/lsdslam>
 *
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/pgo/PGOConfig.h"

#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"

namespace Saiga
{
class SAIGA_VISION_API ArapVertex : public g2o::BaseVertex<6, SE3>
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ArapVertex() { _marginalized = false; }
    virtual bool read(std::istream&) { return true; }
    virtual bool write(std::ostream&) const { return true; }

    virtual void setToOriginImpl() { _estimate = SE3(); }

    virtual void oplusImpl(const double* update_)
    {
        Eigen::Map<Eigen::Matrix<double, 6, 1> > update(const_cast<double*>(update_));

        setEstimate(estimate() * SE3::exp(update));
    }
};


class SAIGA_VISION_API ArapEdgeTarget : public g2o::BaseUnaryEdge<3, Vec3, ArapVertex>
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ArapEdgeTarget() {}

    virtual bool read(std::istream&) { return true; }
    virtual bool write(std::ostream&) const { return false; }

    void computeError()
    {
        SE3 p  = static_cast<const ArapVertex*>(_vertices[0])->estimate();
        Vec3 t = this->measurement();
        _error = p.translation() - t;
        //        _error.setZero();
    }

    void linearizeOplus()
    {
        //        _jacobianOplusXi.setZero();
        _jacobianOplusXi.block<3, 3>(0, 0) = Mat3::Identity();
        _jacobianOplusXi.block<3, 3>(0, 3) = Mat3::Zero();
    }

    virtual void setMeasurement(const Vec3& m) { _measurement = m; }
};



class SAIGA_VISION_API ArapEdge : public g2o::BaseBinaryEdge<3, Vec3, ArapVertex, ArapVertex>
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ArapEdge() {}

    virtual bool read(std::istream&) { return true; }
    virtual bool write(std::ostream&) const { return false; }

    void computeError()
    {
        SE3 pHat  = static_cast<const ArapVertex*>(_vertices[0])->estimate();
        SE3 qHat  = static_cast<const ArapVertex*>(_vertices[1])->estimate();
        Vec3 e_ij = this->measurement();

        Vec3 R_eij = pHat.so3() * e_ij;
        _error     = w_Reg * (pHat.translation() - qHat.translation() - R_eij);
    }

    void linearizeOplus()
    {
        SE3 pHat = static_cast<const ArapVertex*>(_vertices[0])->estimate();
        //        SE3 qHat  = static_cast<const ArapVertex*>(_vertices[1])->estimate();
        Vec3 e_ij = this->measurement();

        Vec3 R_eij = pHat.so3() * e_ij;

        _jacobianOplusXi.block<3, 3>(0, 0) = Mat3::Identity();
        _jacobianOplusXi.block<3, 3>(0, 3) = skew(R_eij);
        _jacobianOplusXi *= w_Reg;

        _jacobianOplusXj.block<3, 3>(0, 0) = -Mat3::Identity();
        _jacobianOplusXj.block<3, 3>(0, 3) = Mat3::Zero();
        _jacobianOplusXj *= w_Reg;
    }

    virtual void setMeasurement(const Vec3& m) { _measurement = m; }

    double w_Reg;
};

}  // namespace Saiga
