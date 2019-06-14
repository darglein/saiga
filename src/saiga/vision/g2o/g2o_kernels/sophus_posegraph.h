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
        setEstimate(estimate() * SE3::exp(update));
#endif
    }
};

/**
 * \brief 7D edge between two Vertex7
 */
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
#ifdef LSD_REL
        SE3 error_ = from.inverse() * to * _inverseMeasurement;
        _error     = error_.log();


//        cout << from << endl;
//        cout << to << endl;
//        cout << _measurement << endl;
//        exit(0);
//        cout << _error.transpose() << endl;

//        SE3 error_2 = _inverseMeasurement * _to->estimate() * _from->estimate().inverse();
//        cout << error_2.log().transpose() << endl;
//        exit(0);
#else
        //        SE3 error_              = _from->estimate() * _to->estimate().inverse() * _measurement;
        SE3 error_ = _measurement * from * to.inverse();
        _error     = error_.log();
//        cout << _error.transpose() << endl;
//        exit(0);
#endif
    }

    void linearizeOplus()
    {
        g2o::BaseBinaryEdge<6, SE3, VertexSim3, VertexSim3>::linearizeOplus();
        return;
#ifdef LSD_REL


        auto from = static_cast<const VertexSim3*>(_vertices[0])->estimate();
        auto to   = static_cast<const VertexSim3*>(_vertices[1])->estimate();


        _jacobianOplusXj = from.inverse().Adj();
        _jacobianOplusXi = -_jacobianOplusXj;

        //        return;
        //        cout << from << endl;
        //        cout << to << endl;
        //        cout << _measurement << endl;
        //        cout << _jacobianOplusXi << endl << endl;
        //        cout << _jacobianOplusXj << endl << endl;

        //        g2o::BaseBinaryEdge<6, SE3, VertexSim3, VertexSim3>::linearizeOplus();

        //        cout << _jacobianOplusXi << endl << endl;
        //        cout << _jacobianOplusXj << endl << endl;

        //        exit(0);
        //        return;
        Eigen::Matrix<double, 6, 6> ji = _jacobianOplusXi;
        Eigen::Matrix<double, 6, 6> jj = _jacobianOplusXj;
        g2o::BaseBinaryEdge<6, SE3, VertexSim3, VertexSim3>::linearizeOplus();

        auto ei = (ji - _jacobianOplusXi).norm();
        auto ej = (jj - _jacobianOplusXj).norm();
        cout << "error i " << ei << endl;
        cout << "error j " << ej << endl;
#else
        const VertexSim3* _from = static_cast<const VertexSim3*>(_vertices[0]);
        const VertexSim3* _to   = static_cast<const VertexSim3*>(_vertices[1]);

        g2o::BaseBinaryEdge<6, SE3, VertexSim3, VertexSim3>::linearizeOplus();
        return;
#    if 0
        cout << "======================================================================================================"
                "============ "
             << endl;
        cout << _from->estimate().inverse().Adj() << endl << endl;
        cout << _from->estimate().Adj() << endl << endl;
        cout << _to->estimate().inverse().Adj() << endl << endl;
        cout << (_from->estimate() * _to->estimate().inverse()).Adj() << endl << endl;
        cout << (_from->estimate() * _to->estimate().inverse()).inverse().Adj() << endl << endl;
        cout << (_measurement * _from->estimate() * _to->estimate().inverse()).Adj() << endl << endl;
        cout << "======================================================================================================"
                "============ "
             << endl;

        cout << "numeric reference:" << endl;
        cout << _jacobianOplusXi << endl << endl;
        cout << _jacobianOplusXj << endl;
#    endif

#    if 1
        cout << "======================================================================================================"
                "============ "
             << endl;
        cout << _from->estimate().inverse().Adj() << endl << endl;
        cout << _from->estimate().Adj() << endl << endl;
        cout << _to->estimate().inverse().Adj() << endl << endl;
        cout << (_from->estimate() * _to->estimate().inverse()).Adj() << endl << endl;
        cout << (_from->estimate() * _to->estimate().inverse()).inverse().Adj() << endl << endl;
        cout << (_from->estimate() * _to->estimate().inverse() * _measurement).Adj() << endl << endl;
        cout << "======================================================================================================"
                "============ "
             << endl;

        cout << "numeric reference:" << endl;
        cout << _jacobianOplusXi << endl << endl;
        cout << _jacobianOplusXj << endl;
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
        cout << "error i " << ei << endl;
        cout << "error j " << ej << endl;

        if (ej > 1)
        {
#    if 0
            cout << "meas:" << endl;
            cout << _measurement.Adj() << endl;
            cout << _measurement << endl;
            cout << Sophus::SO3<double>::hat(_measurement.translation()) << endl;
            cout << "=================================================================================================="
                    "===="
                    "============ ";
                 << endl;
            cout << _from->estimate().inverse().Adj() << endl << endl;
            cout << _from->estimate().Adj() << endl << endl;
            cout << _to->estimate().inverse().Adj() << endl << endl;
            cout << (_from->estimate() * _to->estimate().inverse()).Adj() << endl << endl;
            cout << (_from->estimate() * _to->estimate().inverse()).inverse().Adj() << endl << endl;
            cout << (_measurement * _from->estimate() * _to->estimate().inverse()).Adj() << endl << endl;
            //            cout << (_measurement * _from->estimate() * _to->estimate().inverse()).inverse().Adj() << endl
            //            << endl;

#    endif

#    if 0
            SE3 test = (_from->estimate() * _to->estimate().inverse()).inverse();

            auto R     = test.so3().matrix();
            auto myhat = Sophus::SO3<double>::hat(test.translation());
            cout << test.Adj() << endl << endl;

            auto Rm = _measurement.so3().matrix();
            auto Tm = Sophus::SO3<double>::hat(_measurement.translation());

            cout << myhat * R << endl << endl;
            cout << R * myhat << endl << endl;
            cout << myhat * R * Rm.transpose() << endl << endl;
            cout << myhat * Rm.transpose() * R << endl << endl;
            cout << Rm * myhat * R << endl << endl;
            cout << Rm.transpose() * myhat * R << endl << endl;

            cout << Tm * myhat * R << endl << endl;
            cout << myhat * R * Tm << endl << endl;
            cout << myhat * Tm * R << endl << endl;
#    endif

            cout << "=================================================================================================="
                    "===="
                    "============ "
                 << endl;

            //            cout << _jacobianOplusXi.block(0, 3, 3, 3) * 2 << endl << endl;
            cout << _jacobianOplusXi << endl << endl;
            cout << _jacobianOplusXj << endl << endl;
            cout << _jacobianOplusXi2 << endl << endl;
            cout << _jacobianOplusXj2 << endl << endl;
            exit(0);
        }

        _jacobianOplusXi = _jacobianOplusXi2;
        _jacobianOplusXj = _jacobianOplusXj2;
        //        cout << endl;


        //        //        _jacobianOplusXi        = -_jacobianOplusXj;

        //        cout << endl;
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
