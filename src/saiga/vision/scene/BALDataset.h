/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/scene/Scene.h"



namespace Saiga
{
/**
 * Loads a BAL dataset.
 * http://grail.cs.washington.edu/projects/bal/
 *
 * BAL Camera Model:
 *
 * P  =  R * X + t       (conversion from world to camera coordinates)
 * p  = -P / P[2]         (perspective division)
 * p' =  f * r(p) * p    (conversion to pixel coordinates)
 *
 * where P[2] is the third (z) coordinate of P. In the last equation, r(p) is a function that computes a scaling factor
 * to undo the radial distortion:
 *
 * r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4.
 *
 */
class SAIGA_VISION_API BALDataset
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    struct BALObservation
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int camera_index;
        int point_index;
        Eigen::Vector2d point;

        StereoImagePoint ip()
        {
            StereoImagePoint ip;
            ip.point = point;
            // BAL has the y axis pointing downwards
            ip.point(1) *= -1;
            ip.wp = point_index;
            return ip;
        }
    };
    struct BALCamera
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        SE3 se3;
        double f;
        double k1, k2;

        double r(Vec2 p)
        {
            // 1.0 + k1 * ||p||^2 + k2 * ||p||^4.
            auto r2 = p.squaredNorm();
            return 1.0 + k1 * r2 + k2 * r2 * r2;
        }

        Eigen::Vector2d projectPoint(const Eigen::Vector3d& x)
        {
            Vec3 P = se3 * x;
            Vec2 p = (-P / P(2)).head<2>();
            // This seems to be the actual projection function
            // Maybe a mistake on the webside?
            Vec2 pp = f * p;
            pp      = r(pp) * pp;
            return pp;
        }


        Eigen::Vector2d undistort(const Eigen::Vector2d& x)
        {
            Vec2 p = x;
            for (int j = 0; j < 10; j++)
            {
                p = x / r(p);
            }
            return p;
        }

        IntrinsicsPinholed intr() { return {f, f, 0, 0, 0}; }

        std::pair<SE3, bool> extr()
        {
            // Frome the BAL Docu:
            // the positive x-axis points right, and the positive y-axis points up (in addition, in the camera
            // coordinate system, the positive z-axis points backwards, so the camera is looking down the negative
            // z-axis, as in OpenGL).
            //
            // -> Flip y and z axis
            Mat3 rot;
            rot << 1, 0, 0, 0, -1, 0, 0, 0, -1;
            SE3 res = SE3(Sophus::SO3d::fitToSO3(rot), Vec3(0, 0, 0)) * se3;
            return {res, false};
        }
    };
    struct BALPoint
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Vec3 point;

        WorldPoint wp()
        {
            WorldPoint wp;
            wp.p     = point;
            wp.valid = true;
            return wp;
        }
    };

    BALDataset(const std::string& file);
    void undistortAll();
    double rms();

    Scene makeScene();

   private:
    AlignedVector<BALObservation> observations;
    AlignedVector<BALCamera> cameras;
    AlignedVector<BALPoint> points;
};

}  // namespace Saiga
