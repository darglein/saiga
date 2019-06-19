/**
 * Copyright (c) 2017 Darius Rückert
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
class SAIGA_VISION_API VertexSim3 : public g2o::BaseVertex<6, SE3>
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexSim3() { _marginalized = false; }
    virtual bool read(std::istream&) { return true; }
    virtual bool write(std::ostream&) const { return true; }

    virtual void setToOriginImpl() { _estimate = SE3(); }

    virtual void oplusImpl(const double* update_)
    {
        Eigen::Map<Eigen::Matrix<double, 6, 1> > update(const_cast<double*>(update_));
#ifdef LSD_REL
        setEstimate(SE3::exp(update) * estimate());
#else
        //        setEstimate(estimate() * SE3::exp(update));
        setEstimate(SE3::exp(update) * estimate());
#endif
    }
};

/**
 * \brief 7D edge between two Vertex7
 */
template <bool _LSD_REL = false>
class SAIGA_VISION_API EdgeSim3 : public g2o::BaseBinaryEdge<6, SE3, VertexSim3, VertexSim3>
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSim3() {}

    virtual bool read(std::istream&) { return true; }
    virtual bool write(std::ostream&) const { return false; }

    void computeError()
    {
        auto from = static_cast<const VertexSim3*>(_vertices[0])->estimate();
        auto to   = static_cast<const VertexSim3*>(_vertices[1])->estimate();
        if (_LSD_REL)
        {
            SE3 error_ = from.inverse() * to * _inverseMeasurement;
            _error     = error_.log();


            //        std::cout << from << std::endl;
            //        std::cout << to << std::endl;
            //        std::cout << _measurement << std::endl;
            //        exit(0);
            //        std::cout << _error.transpose() << std::endl;

            //        SE3 error_2 = _inverseMeasurement * _to->estimate() * _from->estimate().inverse();
            //        std::cout << error_2.log().transpose() << std::endl;
            //        exit(0);
        }
        else
        {
            //        SE3 error_              = _from->estimate() * _to->estimate().inverse() * _measurement;
            SE3 error_ = _measurement * from * to.inverse();
            _error     = error_.log();
            //        std::cout << _error.transpose() << std::endl;
            //        exit(0);
        }
    }

    void linearizeOplus()
    {
        if (_LSD_REL)
        {
            auto from = static_cast<const VertexSim3*>(_vertices[0])->estimate();
            //            auto to   = static_cast<const VertexSim3*>(_vertices[1])->estimate();


            _jacobianOplusXj = from.inverse().Adj();
            _jacobianOplusXi = -_jacobianOplusXj;
        }
        else
        {
            g2o::BaseBinaryEdge<6, SE3, VertexSim3, VertexSim3>::linearizeOplus();
        }
        return;
#ifdef LSD_REL



        //        return;
        //        std::cout << from << std::endl;
        //        std::cout << to << std::endl;
        //        std::cout << _measurement << std::endl;
        //        std::cout << _jacobianOplusXi << std::endl << std::endl;
        //        std::cout << _jacobianOplusXj << std::endl << std::endl;

        //        g2o::BaseBinaryEdge<6, SE3, VertexSim3, VertexSim3>::linearizeOplus();

        //        std::cout << _jacobianOplusXi << std::endl << std::endl;
        //        std::cout << _jacobianOplusXj << std::endl << std::endl;

        //        exit(0);
        //        return;
        Eigen::Matrix<double, 6, 6> ji = _jacobianOplusXi;
        Eigen::Matrix<double, 6, 6> jj = _jacobianOplusXj;
        g2o::BaseBinaryEdge<6, SE3, VertexSim3, VertexSim3>::linearizeOplus();

        auto ei = (ji - _jacobianOplusXi).norm();
        auto ej = (jj - _jacobianOplusXj).norm();
        std::cout << "error i " << ei << std::endl;
        std::cout << "error j " << ej << std::endl;
#else
        const VertexSim3* _from = static_cast<const VertexSim3*>(_vertices[0]);
        const VertexSim3* _to   = static_cast<const VertexSim3*>(_vertices[1]);

        g2o::BaseBinaryEdge<6, SE3, VertexSim3, VertexSim3>::linearizeOplus();
        return;
#    if 0
        std::cout << "======================================================================================================"
                "============ "
             << std::endl;
        std::cout << _from->estimate().inverse().Adj() << std::endl << std::endl;
        std::cout << _from->estimate().Adj() << std::endl << std::endl;
        std::cout << _to->estimate().inverse().Adj() << std::endl << std::endl;
        std::cout << (_from->estimate() * _to->estimate().inverse()).Adj() << std::endl << std::endl;
        std::cout << (_from->estimate() * _to->estimate().inverse()).inverse().Adj() << std::endl << std::endl;
        std::cout << (_measurement * _from->estimate() * _to->estimate().inverse()).Adj() << std::endl << std::endl;
        std::cout << "======================================================================================================"
                "============ "
             << std::endl;

        std::cout << "numeric reference:" << std::endl;
        std::cout << _jacobianOplusXi << std::endl << std::endl;
        std::cout << _jacobianOplusXj << std::endl;
#    endif

#    if 1
        std::cout
            << "======================================================================================================"
               "============ "
            << std::endl;
        std::cout << _from->estimate().inverse().Adj() << std::endl << std::endl;
        std::cout << _from->estimate().Adj() << std::endl << std::endl;
        std::cout << _to->estimate().inverse().Adj() << std::endl << std::endl;
        std::cout << (_from->estimate() * _to->estimate().inverse()).Adj() << std::endl << std::endl;
        std::cout << (_from->estimate() * _to->estimate().inverse()).inverse().Adj() << std::endl << std::endl;
        std::cout << (_from->estimate() * _to->estimate().inverse() * _measurement).Adj() << std::endl << std::endl;
        std::cout
            << "======================================================================================================"
               "============ "
            << std::endl;

        std::cout << "numeric reference:" << std::endl;
        std::cout << _jacobianOplusXi << std::endl << std::endl;
        std::cout << _jacobianOplusXj << std::endl;
#    endif

        Eigen::Matrix<double, 6, 6> _jacobianOplusXj2 =
            (_from->estimate() * _to->estimate().inverse() * _measurement).inverse().Adj();
        _jacobianOplusXj2.block(0, 3, 3, 3) *= -0.5;
        _jacobianOplusXj2.block(0, 3, 3, 3).transposeInPlace();

        Eigen::Matrix<double, 6, 6> _jacobianOplusXi2 =
            -((_from->estimate() * _to->estimate().inverse()).inverse()).inverse().Adj();
        _jacobianOplusXi2.block(0, 3, 3, 3) *= 0.5;
        //        _jacobianOplusXi2.block(0, 3, 3, 3).transposeInPlace();

        std::swap(_jacobianOplusXi2, _jacobianOplusXj2);


        _jacobianOplusXi2 = _to->estimate().Adj();
        _jacobianOplusXj2 = -_to->estimate().Adj();

        auto ei = (_jacobianOplusXi2 - _jacobianOplusXi).norm();
        auto ej = (_jacobianOplusXj2 - _jacobianOplusXj).norm();
        std::cout << "error i " << ei << std::endl;
        std::cout << "error j " << ej << std::endl;

        if (ej > 1)
        {
#    if 0
            std::cout << "meas:" << std::endl;
            std::cout << _measurement.Adj() << std::endl;
            std::cout << _measurement << std::endl;
            std::cout << Sophus::SO3<double>::hat(_measurement.translation()) << std::endl;
            std::cout << "=================================================================================================="
                    "===="
                    "============ ";
                 << std::endl;
            std::cout << _from->estimate().inverse().Adj() << std::endl << std::endl;
            std::cout << _from->estimate().Adj() << std::endl << std::endl;
            std::cout << _to->estimate().inverse().Adj() << std::endl << std::endl;
            std::cout << (_from->estimate() * _to->estimate().inverse()).Adj() << std::endl << std::endl;
            std::cout << (_from->estimate() * _to->estimate().inverse()).inverse().Adj() << std::endl << std::endl;
            std::cout << (_measurement * _from->estimate() * _to->estimate().inverse()).Adj() << std::endl << std::endl;
            //            std::cout << (_measurement * _from->estimate() * _to->estimate().inverse()).inverse().Adj() << std::endl
            //            << std::endl;

#    endif

#    if 0
            SE3 test = (_from->estimate() * _to->estimate().inverse()).inverse();

            auto R     = test.so3().matrix();
            auto myhat = Sophus::SO3<double>::hat(test.translation());
            std::cout << test.Adj() << std::endl << std::endl;

            auto Rm = _measurement.so3().matrix();
            auto Tm = Sophus::SO3<double>::hat(_measurement.translation());

            std::cout << myhat * R << std::endl << std::endl;
            std::cout << R * myhat << std::endl << std::endl;
            std::cout << myhat * R * Rm.transpose() << std::endl << std::endl;
            std::cout << myhat * Rm.transpose() * R << std::endl << std::endl;
            std::cout << Rm * myhat * R << std::endl << std::endl;
            std::cout << Rm.transpose() * myhat * R << std::endl << std::endl;

            std::cout << Tm * myhat * R << std::endl << std::endl;
            std::cout << myhat * R * Tm << std::endl << std::endl;
            std::cout << myhat * Tm * R << std::endl << std::endl;
#    endif

            std::cout
                << "=================================================================================================="
                   "===="
                   "============ "
                << std::endl;

            //            std::cout << _jacobianOplusXi.block(0, 3, 3, 3) * 2 << std::endl << std::endl;
            std::cout << _jacobianOplusXi << std::endl << std::endl;
            std::cout << _jacobianOplusXj << std::endl << std::endl;
            std::cout << _jacobianOplusXi2 << std::endl << std::endl;
            std::cout << _jacobianOplusXj2 << std::endl << std::endl;
            exit(0);
        }

        _jacobianOplusXi = _jacobianOplusXi2;
        _jacobianOplusXj = _jacobianOplusXj2;
        //        std::cout << std::endl;


        //        //        _jacobianOplusXi        = -_jacobianOplusXj;

        //        std::cout << std::endl;
        //        exit(0);


#endif
    }


    virtual void setMeasurement(const SE3& m)
    {
        _measurement        = m;
        _inverseMeasurement = m.inverse();
    }

    virtual bool setMeasurementFromState()
    {
        SAIGA_EXIT_ERROR("");
        const VertexSim3* from = static_cast<const VertexSim3*>(_vertices[0]);
        const VertexSim3* to   = static_cast<const VertexSim3*>(_vertices[1]);
        SE3 delta              = from->estimate().inverse() * to->estimate();
        setMeasurement(delta);
        return true;
    }


   protected:
    SE3 _inverseMeasurement;
};

}  // namespace Saiga
