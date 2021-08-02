/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"


namespace Saiga
{
// Based on: Omnidirectional Camera Calibration Toolbox
// https://sites.google.com/site/scarabotix/ocamcalib-toolbox
//
// This is a fisheye camera model, for wide-angle cameras.
//
template <typename T>
struct OCam
{
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using Vec2 = Eigen::Matrix<T, 2, 1>;

    int w, h;
    T c;
    T d;
    T e;
    T cx;
    T cy;
    std::vector<T> poly_cam2world;
    std::vector<T> poly_world2cam;


    // The project/unproject functions are inspired by
    // https://github.com/stonear/OCamCalib/blob/master/ocam_functions.h
    Vec2 Project(Vec3 p)
    {
        T norm  = sqrt(p[0] * p[0] + p[1] * p[1]);
        T theta = atan(p[2] / norm);

        if (norm < 1e-6)
        {
            return Vec2(cx, cy);
        }


        T invnorm = 1 / norm;
        T t       = theta;
        T rho     = poly_world2cam[0];
        int t_i   = 1;

        for (int i = 1; i < poly_world2cam.size(); i++)
        {
            t_i *= t;
            rho += t_i * poly_world2cam[i];
        }

        T x = p[0] * invnorm * rho;
        T y = p[1] * invnorm * rho;

        vec2 res;
        res[0] = x * c + y * d + cx;
        res[1] = x * e + y + cy;
        return res;
    }

    Vec3 InverseProject(Vec2 p, T depth = 1)
    {
        T invdet = 1 / (c - d * e);
        T xp     = invdet * ((p[0] - cx) - d * (p[1] - cy));
        T yp     = invdet * (-e * (p[0] - cx) + c * (p[1] - cy));

        T r   = sqrt(xp * xp + yp * yp);
        T zp  = poly_cam2world[0];
        T r_i = 1;

        for (int i = 1; i < poly_cam2world.size(); i++)
        {
            r_i *= r;
            zp += r_i * poly_cam2world[i];
        }

        Vec3 result(xp, yp, 1);
        return result * depth;
    }
};

template <typename T>
std::ostream& operator<<(std::ostream& strm, const OCam<T> intr)
{
    strm << intr.w << "x" << intr.h << "(" << intr.c << ", " << intr.d << ", " << intr.e << ", " << intr.cx << ", "
         << intr.cy << ") (";
    for (auto d : intr.poly_cam2world)
    {
        strm << d << ", ";
    }
    strm << ") (";
    for (auto d : intr.poly_world2cam)
    {
        strm << d << ", ";
    }
    strm << ")";
    return strm;
}


}  // namespace Saiga
