/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/Core.h"

#include "AutoDiffCodeGen.h"

using namespace Saiga;


class MyScalarCostFunctor
{
   public:
    MyScalarCostFunctor() {}

    template <typename T>
    bool operator()(const T* const _x, const T* const _y, T* _e) const
    {
        using Vec2 = Eigen::Matrix<T, 2, 1>;

        Eigen::Map<const Vec2> x(_x);
        const T& y = *_y;
        Eigen::Map<Vec2> r(_e);

        T d      = x.dot(x);
        Vec2 tmp = x * (T(1.0) / d);
        r        = tmp * sin(y) + x * exp(tmp(1));
        return true;
    }
};
bool Evaluate(double const* const* parameters, double* residuals, double** jacobians)
{
    // This code is generated with ceres::AutoDiffCodeGen
    // See todo/todo.h for more informations.
    const double v_20  = parameters[0][0];
    const double v_24  = parameters[0][1];
    const double v_28  = parameters[1][0];
    const double v_36  = v_20 * v_20;
    const double v_43  = v_20 + v_20;
    const double v_50  = v_24 * v_24;
    const double v_58  = v_24 + v_24;
    const double v_64  = v_36 + v_50;
    const double v_68  = 1.000000;
    const double v_76  = v_68 / v_64;
    const double v_77  = v_76 * v_43;
    const double v_78  = v_76 * v_58;
    const double v_80  = -(v_77);
    const double v_81  = -(v_78);
    const double v_83  = v_80 / v_64;
    const double v_84  = v_81 / v_64;
    const double v_98  = v_20 * v_76;
    const double v_99  = v_20 * v_83;
    const double v_100 = v_20 * v_84;
    const double v_105 = v_99 + v_76;
    const double v_112 = v_24 * v_76;
    const double v_113 = v_24 * v_83;
    const double v_114 = v_24 * v_84;
    const double v_120 = v_114 + v_76;
    const double v_126 = sin(v_28);
    const double v_127 = cos(v_28);
    const double v_135 = exp(v_112);
    const double v_136 = v_135 * v_113;
    const double v_137 = v_135 * v_120;
    const double v_143 = v_98 * v_126;
    const double v_146 = v_98 * v_127;
    const double v_147 = v_105 * v_126;
    const double v_148 = v_100 * v_126;
    const double v_157 = v_20 * v_135;
    const double v_158 = v_20 * v_136;
    const double v_159 = v_20 * v_137;
    const double v_164 = v_158 + v_135;
    const double v_171 = v_143 + v_157;
    const double v_172 = v_147 + v_164;
    const double v_173 = v_148 + v_159;
    const double v_179 = v_112 * v_126;
    const double v_182 = v_112 * v_127;
    const double v_183 = v_113 * v_126;
    const double v_184 = v_120 * v_126;
    const double v_193 = v_24 * v_135;
    const double v_194 = v_24 * v_136;
    const double v_195 = v_24 * v_137;
    const double v_201 = v_195 + v_135;
    const double v_207 = v_179 + v_193;
    const double v_208 = v_183 + v_194;
    const double v_209 = v_184 + v_201;
    residuals[0]       = v_171;
    residuals[1]       = v_207;
    jacobians[0][0]    = v_172;
    jacobians[0][2]    = v_208;
    jacobians[0][1]    = v_173;
    jacobians[0][3]    = v_209;
    jacobians[1][0]    = v_146;
    jacobians[1][1]    = v_182;
    return true;
}
class CostBAMonoAnalytic : public ceres::SizedCostFunction<1, 1>
{
   public:
    using T = double;

    virtual ~CostBAMonoAnalytic() {}

    virtual bool Evaluate(double const* const* _parameters, double* _residuals, double** _jacobians) const
    {
        return true;
    }
};

int main(int argc, char* args[])
{
    std::cout << "autodiff test" << std::endl;

    const int numResiduals     = 2;
    const int numParameters1   = 2;
    const int numParameters2   = 1;
    const int numParametersSum = numParameters1 + numParameters2;


    //    using JetType = ceres::Jet<double, 2>;
    //    JetType J1(3.1, 0);
    //    JetType J2(3.2, 0);
    //    JetType res = f_simple(J1, J2);
    //    std::cout << res.a << " " << res.v.transpose() << std::endl;

    ceres::AutoDiffCodeGen<MyScalarCostFunctor, numResiduals, numParameters1, numParameters2> codeGen(
        new MyScalarCostFunctor());
    codeGen.Generate();


    {
        std::array<double, numParametersSum> params     = {7, 4, 1};
        std::array<double*, numParametersSum> paramsPtr = {&params[0], &params[2]};

        Eigen::Matrix<double, numResiduals, 1> residuals;



        Eigen::Matrix<double, numResiduals, numParameters1, Eigen::RowMajor> J1;
        Eigen::Matrix<double, numResiduals, numParameters2> J2;
        std::array<double*, 2> jacobianPtr = {J1.data(), J2.data()};



        // test result (compare to ceres autodiff)
        auto* cost_function2 =
            new ceres::AutoDiffCostFunction<MyScalarCostFunctor, numResiduals, numParameters1, numParameters2>(
                new MyScalarCostFunctor());
        cost_function2->Evaluate(paramsPtr.data(), residuals.data(), jacobianPtr.data());

        std::cout << "Test. Ceres" << std::endl;
        std::cout << "Residual: " << residuals.transpose() << std::endl;
        std::cout << "Jacobian: " << std::endl << J1 << std::endl;
        std::cout << "Jacobian: " << std::endl << J2 << std::endl;

        J1.setZero();
        J2.setZero();
        Evaluate(paramsPtr.data(), residuals.data(), jacobianPtr.data());

        std::cout << "Test. Our generated functions" << std::endl;

        std::cout << "Residual: " << residuals.transpose() << std::endl;
        std::cout << "Jacobian: " << std::endl << J1 << std::endl;
        std::cout << "Jacobian: " << std::endl << J2 << std::endl;
    }


    return 0;
}
