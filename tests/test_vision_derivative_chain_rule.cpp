/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "gtest/gtest.h"

#include "compare_numbers.h"
#include "numeric_derivative.h"

namespace Saiga
{
template <typename T = double>
HD inline Vector<T, 3> Unproject(const Vector<T, 2>& pixel, const Vector<T, 4>& camera,
                                 Matrix<T, 3, 4>* J_camera = nullptr)
{
    double fx = camera(0);
    double fy = camera(1);
    double cx = camera(2);
    double cy = camera(3);

    Vec3 direction;
    direction(0) = (pixel.x() - cx) / fx;
    direction(1) = (pixel.y() - cy) / fy;
    direction(2) = 1;

    if (J_camera)
    {
        auto& J = *J_camera;
        J.setZero();

        J(0, 0) = -1.0 / (fx * fx) * (pixel.x() - cx);
        J(1, 1) = -1.0 / (fy * fy) * (pixel.y() - cy);

        J(0, 2) = -1.0 / fx;
        J(1, 3) = -1.0 / fy;
    }
    return direction;
}


template <typename T = double>
HD inline double Sqrt(double a, Matrix<T, 1, 1>* jacobian = nullptr)
{
    T result = sqrt(a);

    if (jacobian)
    {
        auto& J = *jacobian;

        J(0, 0) = 1.0 / (2 * sqrt(a));
    }

    return result;
}

HD inline double SelfDot(Vec3 v, Matrix<double, 1, 3>* jacobian = nullptr)
{
    // result = v.dot(v);
    double result = v(0) * v(0) + v(1) * v(1) + v(2) * v(2);

    if (jacobian)
    {
        auto& J = *jacobian;

        J(0, 0) = 2 * v(0);
        J(0, 1) = 2 * v(1);
        J(0, 2) = 2 * v(2);
    }

    return result;
}

template <typename T = double>
HD inline double Norm(const Vector<T, 3>& v, Matrix<T, 1, 3>* J_v = nullptr)
{
    // sqrt(x*x+y*y+z*z)

    Matrix<double, 1, 3> J_a_v;
    auto a = SelfDot(v, &J_a_v);

    Matrix<double, 1, 1> J_result_a;
    auto result = Sqrt(a, &J_result_a);

    if (J_v)
    {
        auto& J = *J_v;

        J = J_result_a * J_a_v;
    }

    return result;
}

HD inline Vec3 DivideVectorByScalar(const Vec3& v, double a, Matrix<double, 3, 3>* J_v = nullptr,
                                    Matrix<double, 3, 1>* J_a = nullptr)
{
    // result_x = v_x / a
    // result_t = v_y / a
    // result_z = v_z / a
    Vec3 result = v / a;

    if (J_v)
    {
        auto& J = *J_v;
        J.setZero();

        J(0, 0) = 1.0 / a;
        J(1, 1) = 1.0 / a;
        J(2, 2) = 1.0 / a;
    }

    if (J_a)
    {
        auto& J = *J_a;

        J(0, 0) = -1.0 / (a * a) * v(0);
        J(1, 0) = -1.0 / (a * a) * v(1);
        J(2, 0) = -1.0 / (a * a) * v(2);
    }
    return result;
}

template <typename T = double>
HD inline Vector<T, 3> Normalize(const Vector<T, 3>& v, Matrix<T, 3, 3>* J_v = nullptr)
{
    Matrix<T, 1, 3> J_l_v;
    T l = Norm(v, &J_l_v);

    Matrix<T, 3, 3> J_result_v;
    Matrix<T, 3, 1> J_result_l;
    Vector<T, 3> result = DivideVectorByScalar(v, l, &J_result_v, &J_result_l);

    if (J_v)
    {
        auto& J = *J_v;

        J = J_result_l * J_l_v;
        J += J_result_v;
    }

    return result;
}

Vec3 RayGeneration(Vec2 uv, Vec4 camera, ivec2 image_size, Matrix<double, 3, 4>* J_camera = nullptr)
{
    Vec2 pixel = uv.array() * (image_size - ivec2(1, 1)).array().cast<double>();

    double fx = camera(0);
    double fy = camera(1);
    double cx = camera(2);
    double cy = camera(3);

    Vec3 direction;
    direction(0) = (pixel.x() - cx) / fx;
    direction(1) = (pixel.y() - cy) / fy;
    direction(2) = 1;


    double length             = direction.norm();
    Vec3 direction_normalized = direction / length;



    direction_normalized(0) =
        direction(0) / sqrt(direction(0) * direction(0) + direction(1) * direction(1) + direction(2) * direction(2));


    double x = pixel.x();
    double y = pixel.y();
    double c = cx;
    double b = cy;
    double f = fx;
    double g = fy;
    direction_normalized(0) =
        ((x - c) / f) / sqrt(((x - c) / f) * ((x - c) / f) + ((y - b) / g) * ((y - b) / g) + 1 * 1);

    if (J_camera)
    {
        auto& J = *J_camera;
        J.setZero();

        J(0, 0) = -1.0 / (fx * fx) * (pixel.x() - cx);
        J(1, 1) = -1.0 / (fy * fy) * (pixel.y() - cy);

        J(0, 2) = -1.0 / fx;
        J(1, 3) = -1.0 / fy;
    }

    return direction_normalized;
}


Vec3 RayGenerationChainRule(Vec2 uv, Vec4 camera, ivec2 image_size, Matrix<double, 3, 4>* J_camera = nullptr)
{
    Vec2 pixel = uv.array() * (image_size - ivec2(1, 1)).array().cast<double>();

    Matrix<double, 3, 4> J_direction_camera;
    Vec3 direction = Unproject(pixel, camera, &J_direction_camera);

    Matrix<double, 3, 3> J_result_direction;
    Vec3 result = Normalize(direction, &J_result_direction);

    if (J_camera)
    {
        auto& J = *J_camera;
        J       = J_result_direction * J_direction_camera;
    }

    return result;
}


TEST(NumericDerivative, RayGeneration)
{
    Vec2 uv     = Random::MatrixUniform<Vec2>(0, 1);
    Vec4 camera = Random::MatrixUniform<Vec4>(100, 200);
    ivec2 image_size(500, 500);

    Matrix<double, 3, 4> J_camera, J_camera_numeric;

    auto res1 = RayGenerationChainRule(uv, camera, image_size, &J_camera);
    auto res2 = EvaluateNumeric([&](auto p) { return RayGeneration(uv, p, image_size); }, camera, &J_camera_numeric);

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_camera, J_camera_numeric, 1e-5);
}



}  // namespace Saiga
