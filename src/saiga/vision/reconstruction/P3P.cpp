/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "P3P.h"

#include "saiga/core/math/Polynomial4.h"
#include "saiga/vision/icp/ICPAlign.h"

namespace Saiga
{
inline Vec3 LiftImagePoint(const Vec2& point)
{
    return point.homogeneous() / std::sqrt(point.squaredNorm() + 1);
}


int P3P::solve(ArrayView<const Vec3> worldPoints, ArrayView<const Vec2> normalizedImagePoints,
               std::array<SE3, 4>& results)
{
    SAIGA_ASSERT(worldPoints.size() == 3);
    SAIGA_ASSERT(normalizedImagePoints.size() == 3);

    Eigen::Matrix3d worldPoints_world;
    worldPoints_world.col(0) = worldPoints[0];
    worldPoints_world.col(1) = worldPoints[1];
    worldPoints_world.col(2) = worldPoints[2];

    // The 3D direction of the homogeneous point
    const Eigen::Vector3d u = LiftImagePoint(normalizedImagePoints[0]);
    const Eigen::Vector3d v = LiftImagePoint(normalizedImagePoints[1]);
    const Eigen::Vector3d w = LiftImagePoint(normalizedImagePoints[2]);

    // Angles between 2D points.
    const double cos_uv = u.transpose() * v;
    const double cos_uw = u.transpose() * w;
    const double cos_vw = v.transpose() * w;

    // Distances between 2D points.
    const double dist_AB_2 = (worldPoints[0] - worldPoints[1]).squaredNorm();
    const double dist_AC_2 = (worldPoints[0] - worldPoints[2]).squaredNorm();
    const double dist_BC_2 = (worldPoints[1] - worldPoints[2]).squaredNorm();

    const double dist_AB = std::sqrt(dist_AB_2);

    const double a = dist_BC_2 / dist_AB_2;
    const double b = dist_AC_2 / dist_AB_2;

    // Helper variables for calculation of coefficients.
    const double a2 = a * a;
    const double b2 = b * b;
    const double p  = 2 * cos_vw;
    const double q  = 2 * cos_uw;
    const double r  = 2 * cos_uv;
    const double p2 = p * p;
    const double p3 = p2 * p;
    const double q2 = q * q;
    const double r2 = r * r;
    const double r3 = r2 * r;
    const double r4 = r3 * r;
    const double r5 = r4 * r;


    // Build polynomial coefficients: a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0 = 0.
    Eigen::Matrix<double, 5, 1> coeffs;
    coeffs(0) = -2 * b + b2 + a2 + 1 + a * b * (2 - r2) - 2 * a;

    // Check reality condition
    if (coeffs(0) == 0) return 0;


    coeffs(1) = -2 * q * a2 - r * p * b2 + 4 * q * a + (2 * q + p * r) * b + (r2 * q - 2 * q + r * p) * a * b - 2 * q;
    coeffs(2) = (2 + q2) * a2 + (p2 + r2 - 2) * b2 - (4 + 2 * q2) * a - (p * q * r + p2) * b -
                (p * q * r + r2) * a * b + q2 + 2;
    coeffs(3) = -2 * q * a2 - r * p * b2 + 4 * q * a + (p * r + q * p2 - 2 * q) * b + (r * p + 2 * q) * a * b - 2 * q;
    coeffs(4) = a2 + b2 - 2 * a + (2 - p2) * b - 2 * a * b + 1;


    Vec4 roots_real;
    int num_real_roots = solve_deg4(coeffs, roots_real);

    std::array<SE3, 4> models;

    int actualSolutions = 0;
    for (int i = 0; i < num_real_roots; ++i)
    {
        const double x = roots_real(i);
        if (x <= 0)
        {
            continue;
        }

        const double x2 = x * x;
        const double x3 = x2 * x;

        // Build polynomial coefficients: b1*y + b0 = 0.
        const double bb1 = (p2 - p * q * r + r2) * a + (p2 - r2) * b - p2 + p * q * r - r2;
        const double b1  = b * bb1 * bb1;
        const double b0 =
            ((1 - a - b) * x2 + (a - 1) * q * x - a + b + 1) *
            (r3 * (a2 + b2 - 2 * a - 2 * b + (2 - r2) * a * b + 1) * x3 +
             r2 *
                 (p + p * a2 - 2 * r * q * a * b + 2 * r * q * b - 2 * r * q - 2 * p * a - 2 * p * b + p * r2 * b +
                  4 * r * q * a + q * r3 * a * b - 2 * r * q * a2 + 2 * p * a * b + p * b2 - r2 * p * b2) *
                 x2 +
             (r5 * (b2 - a * b) - r4 * p * q * b + r3 * (q2 - 4 * a - 2 * q2 * a + q2 * a2 + 2 * a2 - 2 * b2 + 2) +
              r2 * (4 * p * q * a - 2 * p * q * a * b + 2 * p * q * b - 2 * p * q - 2 * p * q * a2) +
              r * (p2 * b2 - 2 * p2 * b + 2 * p2 * a * b - 2 * p2 * a + p2 + p2 * a2)) *
                 x +
             (2 * p * r2 - 2 * r3 * q + p3 - 2 * p2 * q * r + p * q2 * r2) * a2 + (p3 - 2 * p * r2) * b2 +
             (4 * q * r3 - 4 * p * r2 - 2 * p3 + 4 * p2 * q * r - 2 * p * q2 * r2) * a +
             (-2 * q * r3 + p * r4 + 2 * p2 * q * r - 2 * p3) * b + (2 * p3 + 2 * q * r3 - 2 * p2 * q * r) * a * b +
             p * q2 * r2 - 2 * p2 * q * r + 2 * p * r2 + p3 - 2 * r3 * q);



        // Check reality condition
        if (b0 <= 0) continue;

        // Solve for y.
        const double y  = b0 / b1;
        const double y2 = y * y;

        const double nu = x2 + y2 - 2 * x * y * cos_uv;

        if (nu <= 0) continue;

        const double dist_PC = dist_AB / std::sqrt(nu);
        const double dist_PB = y * dist_PC;
        const double dist_PA = x * dist_PC;

        Eigen::Matrix3d worldPoints_camera;
        worldPoints_camera.col(0) = u * dist_PA;  // A'
        worldPoints_camera.col(1) = v * dist_PB;  // B'
        worldPoints_camera.col(2) = w * dist_PC;  // C'

        // Find transformation from the world to the camera system.
#if 0
        const Eigen::Matrix4d transform = Eigen::umeyama(worldPoints_world, worldPoints_camera, false);
        SE3 se3                         = SE3::fitToSE3(transform);
#else
        SE3 se3 = ICP::alignMinimal(worldPoints_world, worldPoints_camera);
#endif
        models[actualSolutions++] = se3;
    }

    results = models;
    return actualSolutions;
}

std::optional<SE3> P3P::bestSolution(ArrayView<const SE3> candidates, const Vec3& fourthWorldPoint,
                                     const Vec2& fourthImagePoint)
{
    std::optional<SE3> result;

    int ns            = -1;
    double min_reproj = std::numeric_limits<double>::max();

    // Pick solution with smallest reprojection error and positive z
    for (int i = 0; i < (int)candidates.size(); i++)
    {
        Vec3 p   = candidates[i] * fourthWorldPoint;
        double z = p.z();
        if (z <= 0) continue;
        p          = p / z;
        auto error = (p.head<2>() - fourthImagePoint).squaredNorm();
        if (error < min_reproj)
        {
            ns         = i;
            min_reproj = error;
        }
    }

    if (ns != -1) result = candidates[ns];

    return result;
}

std::optional<SE3> P3P::solve4(ArrayView<const Vec3> worldPoints, ArrayView<const Vec2> normalizedImagePoints)
{
    SAIGA_ASSERT(worldPoints.size() == 4);
    std::array<SE3, 4> multiResults;
    int n = solve(worldPoints.head(3), normalizedImagePoints.head(3), multiResults);

    //    return SE3();
    return bestSolution(ArrayView<SE3>(multiResults).head(n), worldPoints[3], normalizedImagePoints[3]);
}

int P3PRansac::solve(ArrayView<const Vec3> _worldPoints, ArrayView<const Vec2> _normalizedImagePoints, SE3& bestT,
                     std::vector<int>& bestInlierMatches, std::vector<char>& inlierMask)
{
#pragma omp single
    {
        worldPoints           = _worldPoints;
        normalizedImagePoints = _normalizedImagePoints;
        N                     = _worldPoints.size();
    }


    int idx;
    idx = compute(_worldPoints.size());

#pragma omp single
    {
        bestT      = models[idx];
        inlierMask = inliers[idx];

        bestInlierMatches.clear();
        bestInlierMatches.reserve(numInliers[idx]);
        for (int i = 0; i < N; ++i)
        {
            if (inliers[idx][i]) bestInlierMatches.push_back(i);
        }
    }

    return numInliers[idx];
}

bool P3PRansac::computeModel(const RansacBase::Subset& set, P3PRansac::Model& model)
{
    std::array<Vec3, 4> A;
    std::array<Vec2, 4> B;

    for (auto i : Range(0, (int)set.size()))
    {
        A[i] = worldPoints[set[i]];
        B[i] = normalizedImagePoints[set[i]];
    }

    auto res = P3P::solve4(A, B);


    if (res.has_value())
    {
        model = res.value();
        return true;
    }

    return false;
}

double P3PRansac::computeResidual(const P3PRansac::Model& model, int i)
{
    Vec2 ip = (model * worldPoints[i]).hnormalized();
    return (ip - normalizedImagePoints[i]).squaredNorm();
}


#if 0
SE3 refinePose(const SE3& pose, const Vec3* worldPoints, const Vec2* normalizedImagePoints, int N, int iterations)
{
    using MonoKernel = typename Saiga::Kernel::BAPoseMono<T, false, true>;
    using JType      = Eigen::Matrix<T, 6, 6>;
    using BType      = Eigen::Matrix<T, 6, 1>;
    typename MonoKernel::CameraType dummy;

    typename MonoKernel::JacobiType JrowM;
    JType JtJ;
    BType Jtb;

    SE3 guess = pose;
    for (auto it : Range(0, iterations))
    {
        JtJ.setZero();
        Jtb.setZero();
        double chi2sum = 0;

        for (auto i : Range(0, N))
        {
            auto&& wp = worldPoints[i];
            auto&& ip = normalizedImagePoints[i];

            Vec2 res;
            MonoKernel::evaluateResidualAndJacobian(dummy, guess, wp, ip, res, JrowM, 1);
                auto c2 = res.squaredNorm();

                chi2sum += c2;
                JtJ += (JrowM.transpose() * JrowM);
                Jtb += JrowM.transpose() * res;
            }

            BType x = JtJ.ldlt().solve(Jtb);
            guess   = SE3Type::exp(x) * guess;

            //            std::cout << "chi2 " << chi2sum << std::endl;
        }
        return guess;
    }
#endif
}  // namespace Saiga
